const DEFAULT_UPSTREAM_BASE_URL = "https://unlimited.surf";
const DEFAULT_OPENAI_MODEL = "gateway-gpt-5-5";
const DEFAULT_CLAUDE_MODEL = "claude-opus-4-7-20260101";
const TOOL_CALL_SENTINEL = "__TRANSFER_API_TOOL_CALLS__";

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
  "Access-Control-Allow-Headers": "authorization,content-type,x-api-key,anthropic-api-key,anthropic-version,anthropic-beta,openai-beta",
  "Access-Control-Expose-Headers": "content-type,request-id,x-request-id",
};

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    const url = new URL(request.url);
    const path = normalizePath(url.pathname);

    try {
      const authError = validateWorkerApiKey(request, env);
      if (authError) return authError;

      if (path === "/" || path === "/health") {
        return jsonResponse(serviceInfo(request, env));
      }

      if (path.startsWith("/api/")) {
        return proxyUpstream(request, env, path);
      }

      if (path === "/mcp" || path === "/v1/mcp" || path === "/anthropic/mcp" || path === "/anthropic/v1/mcp") {
        return jsonResponse(mcpInfo(request));
      }

      if (path === "/codex" || path === "/v1/codex" || path === "/anthropic/codex" || path === "/anthropic/v1/codex") {
        return textResponse(codexSetup(request), "text/plain; charset=utf-8");
      }

      if (path === "/v1/setup" || path === "/anthropic/setup" || path === "/anthropic/v1/setup") {
        return textResponse(agentSetup(request, env), "text/plain; charset=utf-8");
      }

      if (path === "/v1/messages" || (path === "/v1/models" && looksLikeAnthropicRequest(request)) || path.startsWith("/anthropic/")) {
        return handleAnthropic(request, env, path);
      }

      if (path.startsWith("/v1/")) {
        return handleOpenAI(request, env, path);
      }

      return errorResponse(404, "not_found", `No route for ${path}`);
    } catch (error) {
      return errorResponse(500, "internal_error", error && error.message ? error.message : String(error));
    }
  },
};

async function handleOpenAI(request, env, path) {
  if ((path === "/v1/key" || path === "/v1/auth-key" || path === "/v1/usage") && request.method === "GET") {
    const rawPath = path === "/v1/usage" ? "/api/usage" : "/api/key";
    return proxyUpstream(request, env, rawPath);
  }

  if (path === "/v1/models" && request.method === "GET") {
    if (looksLikeAnthropicRequest(request)) {
      return anthropicModels(request, env);
    }
    return openAIModels(request, env);
  }

  if (path === "/v1/search" && request.method === "POST") {
    const body = await readJson(request);
    return openAIDirectCapability(request, env, body, "/api/search");
  }

  if (path === "/v1/merge" && request.method === "POST") {
    const body = await readJson(request);
    return openAIDirectCapability(request, env, body, "/api/merge");
  }

  if (path === "/v1/chat/completions" && request.method === "POST") {
    const body = await readJson(request);
    return openAIChatCompletions(request, env, body);
  }

  if (path === "/v1/responses" && request.method === "POST") {
    const body = await readJson(request);
    return openAIResponses(request, env, body);
  }

  if (path === "/v1/files" && request.method === "GET") {
    return jsonResponse({ object: "list", data: [], has_more: false });
  }

  if (path === "/v1/files" && request.method === "POST") {
    return openAIFileUpload(request, env);
  }

  if ((path === "/v1/files/extract" || path === "/v1/attachments/extract") && request.method === "POST") {
    const body = await readJson(request);
    const extracted = await callUnlimitedJson(request, env, "/api/attachments/extract", body);
    return jsonResponse(extracted);
  }

  if (path.startsWith("/v1/files/") && request.method === "GET") {
    return errorResponse(404, "not_found", "This Worker is stateless. Bind KV/R2 if you need persisted OpenAI file retrieval.");
  }

  if (path === "/v1/embeddings" || path.startsWith("/v1/audio/") || path.startsWith("/v1/images/")) {
    return errorResponse(501, "unsupported_endpoint", `${path} is not exposed by unlimited.surf and cannot be emulated faithfully.`);
  }

  return errorResponse(404, "not_found", `Unsupported OpenAI-compatible route ${path}`);
}

async function openAIDirectCapability(request, env, body, route) {
  const model = body.model || env.DEFAULT_MODEL || DEFAULT_OPENAI_MODEL;
  const created = nowSeconds();
  const id = `chatcmpl_${randomId()}`;
  const payload = buildUnlimitedPayload({ ...body, web_search: route === "/api/search", merge: route === "/api/merge" }, route);

  if (body.stream !== false) {
    const upstream = await callUnlimitedStream(request, env, route, payload);
    return sseResponse(streamOpenAIChat(upstream, { id, created, model }));
  }

  const result = await collectUnlimitedText(request, env, route, payload);
  return jsonResponse({
    id,
    object: "chat.completion",
    created,
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: result.text },
        logprobs: null,
        finish_reason: result.finishReason || "stop",
      },
    ],
    usage: usageFromText(payload.message || payload.query || "", result.text),
    system_fingerprint: `unlimited-surf-worker:${route}`,
  });
}

async function openAIChatCompletions(request, env, body) {
  const model = body.model || env.DEFAULT_MODEL || DEFAULT_OPENAI_MODEL;
  const created = nowSeconds();
  const id = `chatcmpl_${randomId()}`;
  const toolContext = openAIToolContext(body);

  if (toolContext.enabled) {
    const bridge = await openAIChatViaAnthropic(request, env, body, toolContext);
    if (body.stream) return sseResponse(streamOpenAIChatBridge(bridge, { id, created, model }));
    return jsonResponse(openAIChatBridgeResponse(bridge, { id, created, model }));
  }

  const route = chooseUnlimitedRoute(body);
  const payload = buildUnlimitedPayload(withToolInstructions(body, toolContext, "openai"), route);

  if (body.stream) {
    const upstream = await callUnlimitedStream(request, env, route, payload);
    return sseResponse(streamOpenAIChat(upstream, { id, created, model }));
  }

  const result = await collectUnlimitedText(request, env, route, payload);
  const toolCalls = parseToolCalls(result.text, toolContext);
  if (toolCalls.length) {
    return jsonResponse({
      id,
      object: "chat.completion",
      created,
      model,
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: null, tool_calls: toolCalls.map(toOpenAIChatToolCall) },
          logprobs: null,
          finish_reason: "tool_calls",
        },
      ],
      usage: usageFromText(payload.message || "", result.text),
      system_fingerprint: "unlimited-surf-worker:tool-adapter",
    });
  }

  return jsonResponse({
    id,
    object: "chat.completion",
    created,
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: result.text },
        logprobs: null,
        finish_reason: result.finishReason || "stop",
      },
    ],
    usage: usageFromText(payload.message || "", result.text),
    system_fingerprint: "unlimited-surf-worker",
  });
}

async function openAIResponses(request, env, body) {
  const model = body.model || env.DEFAULT_MODEL || DEFAULT_OPENAI_MODEL;
  const created = nowSeconds();
  const id = `resp_${randomId()}`;
  const toolContext = openAIResponsesToolContext(body);

  if (toolContext.enabled) {
    const bridge = await openAIResponsesViaAnthropic(request, env, body, toolContext);
    if (body.stream) return sseResponse(streamOpenAIResponsesBridge(bridge, { id, created, model }));
    return jsonResponse(openAIResponsesBridgeResponse(body, bridge, { id, created, model }));
  }

  const syntheticChatBody = responsesToChatBody(body, model);
  const route = chooseUnlimitedRoute(syntheticChatBody);
  const payload = buildUnlimitedPayload(withToolInstructions(syntheticChatBody, toolContext, "openai-responses"), route);

  if (body.stream) {
    const upstream = await callUnlimitedStream(request, env, route, payload);
    return sseResponse(streamOpenAIResponses(upstream, { id, created, model }));
  }

  const result = await collectUnlimitedText(request, env, route, payload);
  const toolCalls = parseToolCalls(result.text, toolContext);
  if (toolCalls.length) {
    return jsonResponse(openAIResponseEnvelope(body, {
      id,
      created,
      model,
      output: toolCalls.map(toOpenAIResponseToolCall),
      outputText: "",
      status: "completed",
      usageInput: payload.message || "",
      usageOutput: result.text,
    }));
  }

  return jsonResponse(openAIResponseEnvelope(body, {
    id,
    created,
    model,
    output: [
      {
        id: `msg_${randomId()}`,
        type: "message",
        status: "completed",
        role: "assistant",
        content: [{ type: "output_text", text: result.text, annotations: [] }],
      },
    ],
    outputText: result.text,
    status: "completed",
    usageInput: payload.message || "",
    usageOutput: result.text,
  }));
}

async function handleAnthropic(request, env, path) {
  const anthPath = path.startsWith("/anthropic/") ? normalizePath(path.slice("/anthropic".length) || "/") : path;

  if ((anthPath === "/v1/key" || anthPath === "/key" || anthPath === "/v1/auth-key" || anthPath === "/auth-key") && request.method === "GET") {
    return proxyUpstream(request, env, "/api/key");
  }

  if ((anthPath === "/v1/usage" || anthPath === "/usage") && request.method === "GET") {
    return proxyUpstream(request, env, "/api/usage");
  }

  if ((anthPath === "/v1/models" || anthPath === "/models") && request.method === "GET") {
    return anthropicModels(request, env);
  }

  if ((anthPath === "/v1/messages" || anthPath === "/messages") && request.method === "POST") {
    const body = await readJson(request);
    return upstreamAnthropicMessages(request, env, body);
  }

  if ((anthPath === "/v1/search" || anthPath === "/search") && request.method === "POST") {
    const body = await readJson(request);
    return anthropicDirectCapability(request, env, body, "/api/search");
  }

  if ((anthPath === "/v1/merge" || anthPath === "/merge") && request.method === "POST") {
    const body = await readJson(request);
    return anthropicDirectCapability(request, env, body, "/api/merge");
  }

  if (anthPath === "/v1/setup" || anthPath === "/setup") {
    return textResponse(agentSetup(request, env), "text/plain; charset=utf-8");
  }

  return errorResponse(404, "not_found", `Unsupported Anthropic-compatible route ${path}`);
}

async function anthropicDirectCapability(request, env, body, route) {
  const requestedModel = body.model || env.DEFAULT_CLAUDE_MODEL || DEFAULT_CLAUDE_MODEL;
  const payload = buildAnthropicUnlimitedPayload({ ...body, web_search: route === "/api/search", merge: route === "/api/merge" }, route);
  const id = `msg_${randomId()}`;

  if (body.stream !== false) {
    const upstream = await callUnlimitedStream(request, env, route, payload);
    return sseResponse(streamAnthropicMessages(upstream, { id, model: requestedModel }));
  }

  const result = await collectUnlimitedText(request, env, route, payload);
  return jsonResponse({
    id,
    type: "message",
    role: "assistant",
    model: requestedModel,
    content: [{ type: "text", text: result.text }],
    stop_reason: anthropicStopReason(result.finishReason),
    stop_sequence: null,
    usage: anthropicUsageFromText(payload.message || payload.query || "", result.text),
  });
}

async function upstreamAnthropicMessages(request, env, body) {
  const payload = normalizeAnthropicRequestBody(body, env);
  const response = await fetch(new URL("/v1/messages", upstreamBase(env)), {
    method: "POST",
    headers: anthropicUpstreamHeaders(request, env, Boolean(payload.stream)),
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`upstream /v1/messages failed: ${response.status} ${detail}`);
  }

  return addCors(response);
}

async function anthropicMessages(request, env, body) {
  const requestedModel = body.model || env.DEFAULT_CLAUDE_MODEL || DEFAULT_CLAUDE_MODEL;
  const route = chooseUnlimitedRoute(body);
  const toolContext = anthropicToolContext(body);
  const payload = buildAnthropicUnlimitedPayload(withToolInstructions(body, toolContext, "anthropic"), route);
  const id = `msg_${randomId()}`;

  if (body.stream) {
    const upstream = await callUnlimitedStream(request, env, route, payload);
    return sseResponse(streamAnthropicMessages(upstream, { id, model: requestedModel }));
  }

  const result = await collectUnlimitedText(request, env, route, payload);
  const toolCalls = parseToolCalls(result.text, toolContext);
  if (toolCalls.length) {
    return jsonResponse({
      id,
      type: "message",
      role: "assistant",
      model: requestedModel,
      content: toolCalls.map(toAnthropicToolUse),
      stop_reason: "tool_use",
      stop_sequence: null,
      usage: anthropicUsageFromText(payload.message || "", result.text),
    });
  }

  return jsonResponse({
    id,
    type: "message",
    role: "assistant",
    model: requestedModel,
    content: [{ type: "text", text: result.text }],
    stop_reason: anthropicStopReason(result.finishReason),
    stop_sequence: null,
    usage: anthropicUsageFromText(payload.message || "", result.text),
  });
}

async function openAIModels(request, env) {
  const catalog = await getModelCatalog(request, env);
  return jsonResponse({
    object: "list",
    data: catalog.map((model) => ({
      id: model.id,
      object: "model",
      created: 0,
      owned_by: model.provider || "unlimited.surf",
      permission: [],
      root: model.id,
      parent: null,
    })),
  });
}

async function anthropicModels(request, env) {
  const catalog = await getModelCatalog(request, env);
  const claudeModels = catalog
    .filter((model) => /claude|anthropic/i.test(`${model.id} ${model.name || ""} ${model.provider || ""}`))
    .map((model) => toAnthropicModel(model));

  return jsonResponse({
    data: claudeModels.length ? claudeModels : [toAnthropicModel({ id: DEFAULT_CLAUDE_MODEL, name: "Claude Opus 4.7" })],
    has_more: false,
    first_id: claudeModels[0] ? claudeModels[0].id : DEFAULT_CLAUDE_MODEL,
    last_id: claudeModels[claudeModels.length - 1] ? claudeModels[claudeModels.length - 1].id : DEFAULT_CLAUDE_MODEL,
  });
}

async function openAIFileUpload(request, env) {
  const contentType = request.headers.get("content-type") || "";
  if (!contentType.includes("multipart/form-data")) {
    return errorResponse(400, "invalid_request_error", "OpenAI file upload expects multipart/form-data with a file field.");
  }

  const form = await request.formData();
  const file = form.get("file");
  if (!file || typeof file === "string") {
    return errorResponse(400, "invalid_request_error", "Missing multipart file field named file.");
  }

  const bytes = new Uint8Array(await file.arrayBuffer());
  const payload = {
    name: file.name || "upload.bin",
    type: file.type || "application/octet-stream",
    data: bytesToBase64(bytes),
  };
  const extracted = await callUnlimitedJson(request, env, "/api/attachments/extract", payload);
  const id = `file_${randomId()}`;
  return jsonResponse({
    id,
    object: "file",
    bytes: bytes.byteLength,
    created_at: nowSeconds(),
    filename: payload.name,
    purpose: form.get("purpose") || "assistants",
    status: extracted && extracted.success === false ? "error" : "processed",
    status_details: null,
    unlimited_extract: extracted,
  });
}

function chooseUnlimitedRoute(body) {
  if (body.models && Array.isArray(body.models) && body.models.length >= 2) return "/api/merge";
  if (body.merge || body.merge_ai) return "/api/merge";
  if (body.query || body.web_search || body.web_search_options || hasWebSearchTool(body.tools)) return "/api/search";
  return "/api/chat";
}

function buildUnlimitedPayload(body, route) {
  if (route === "/api/search") {
    return {
      query: body.query || latestUserText(body.messages) || inputToText(body.input) || body.prompt || "",
      model: mapUpstreamModel(body.model),
      effort: body.effort || reasoningEffort(body),
    };
  }

  const message = body.message || messagesToText(body.messages) || inputToText(body.input) || body.prompt || "";
  const payload = {
    message,
    model: mapUpstreamModel(body.model),
    effort: body.effort || reasoningEffort(body),
  };

  if (route === "/api/merge") {
    payload.models = Array.isArray(body.models) && body.models.length ? body.models.map(mapUpstreamModel) : undefined;
  }

  return payload;
}

function buildAnthropicUnlimitedPayload(body, route) {
  if (route === "/api/search") {
    return {
      query: latestUserText(body.messages) || body.query || "",
      model: mapUpstreamModel(body.model),
      effort: body.effort || reasoningEffort(body),
    };
  }

  const prompt = anthropicMessagesToText(body);
  const payload = {
    message: prompt,
    model: mapUpstreamModel(body.model),
    effort: body.effort || reasoningEffort(body),
  };

  if (route === "/api/merge") {
    payload.models = Array.isArray(body.models) && body.models.length ? body.models.map(mapUpstreamModel) : undefined;
  }

  return payload;
}

function openAIToolContext(body) {
  const tools = Array.isArray(body.tools)
    ? body.tools.map((tool) => {
      if (tool && tool.type === "function" && tool.function) {
        return {
          name: tool.function.name,
          description: tool.function.description || "",
          parameters: tool.function.parameters || { type: "object", properties: {} },
        };
      }
      if (tool && tool.function) {
        return {
          name: tool.function.name,
          description: tool.function.description || "",
          parameters: tool.function.parameters || { type: "object", properties: {} },
        };
      }
      return normalizeToolDefinition(tool);
    })
    : [];
  return buildToolContext(tools, body.tool_choice);
}

function openAIResponsesToolContext(body) {
  const tools = Array.isArray(body.tools)
    ? body.tools.map((tool) => {
      if (tool && tool.type === "function") {
        return {
          name: tool.name,
          description: tool.description || "",
          parameters: tool.parameters || { type: "object", properties: {} },
        };
      }
      return normalizeToolDefinition(tool);
    })
    : [];
  return buildToolContext(tools, body.tool_choice);
}

function anthropicToolContext(body) {
  const tools = Array.isArray(body.tools)
    ? body.tools.map((tool) => ({
      name: tool.name,
      description: tool.description || "",
      parameters: tool.input_schema || tool.parameters || { type: "object", properties: {} },
    }))
    : [];
  return buildToolContext(tools, body.tool_choice);
}

function normalizeToolDefinition(tool) {
  if (!tool || typeof tool !== "object") return null;
  return {
    name: tool.name || tool.type,
    description: tool.description || "",
    parameters: tool.parameters || tool.input_schema || { type: "object", properties: {} },
  };
}

function buildToolContext(tools, toolChoice) {
  const normalized = tools
    .filter((tool) => tool && tool.name)
    .map((tool) => ({
      name: String(tool.name),
      description: tool.description || "",
      parameters: tool.parameters || { type: "object", properties: {} },
    }));

  return {
    tools: normalized,
    toolChoice: toolChoice || "auto",
    enabled: normalized.length > 0 && toolChoice !== "none",
  };
}

function withToolInstructions(body, toolContext, protocol) {
  if (!toolContext.enabled) return body;
  const toolInstruction = toolAdapterInstruction(toolContext, protocol);

  if (Array.isArray(body.messages)) {
    return {
      ...body,
      messages: [
        { role: "system", content: toolInstruction },
        ...body.messages,
      ],
    };
  }

  if (Array.isArray(body.input)) {
    return {
      ...body,
      input: [
        { role: "system", content: [{ type: "input_text", text: toolInstruction }] },
        ...body.input,
      ],
    };
  }

  if (typeof body.input === "string") {
    return { ...body, input: `${toolInstruction}\n\n${body.input}` };
  }

  return { ...body, prompt: `${toolInstruction}\n\n${body.prompt || body.message || ""}`.trim() };
}

function toolAdapterInstruction(toolContext, protocol) {
  return [
    "You may call tools provided by the client agent.",
    "When a tool is needed, do not explain it in prose.",
    `Return only one JSON object on a single line in this exact shape: {"${TOOL_CALL_SENTINEL}":[{"name":"tool_name","arguments":{}}]}`,
    "The arguments object must follow the selected tool schema. Multiple tool calls are allowed.",
    `Tool choice: ${JSON.stringify(toolContext.toolChoice)}.`,
    `Client protocol: ${protocol}.`,
    `Available tools: ${JSON.stringify(toolContext.tools)}.`,
    "If no tool is needed, answer normally.",
  ].join("\n");
}

function parseToolCalls(text, toolContext) {
  if (!toolContext.enabled || !text) return [];
  const parsed = parseToolCallJson(text);
  if (!parsed) return [];
  const rawCalls = Array.isArray(parsed[TOOL_CALL_SENTINEL])
    ? parsed[TOOL_CALL_SENTINEL]
    : Array.isArray(parsed.tool_calls)
      ? parsed.tool_calls
      : Array.isArray(parsed.tools)
        ? parsed.tools
        : parsed.name
          ? [parsed]
          : [];
  const names = new Set(toolContext.tools.map((tool) => tool.name));

  return rawCalls
    .map((call) => normalizeToolCall(call))
    .filter((call) => call && names.has(call.name));
}

function parseToolCallJson(text) {
  const trimmed = String(text).trim();
  const candidates = [trimmed];
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced) candidates.push(fenced[1].trim());
  const sentinelIndex = trimmed.indexOf(TOOL_CALL_SENTINEL);
  if (sentinelIndex >= 0) {
    const start = trimmed.lastIndexOf("{", sentinelIndex);
    const end = trimmed.lastIndexOf("}");
    if (start >= 0 && end > start) candidates.push(trimmed.slice(start, end + 1));
  }

  for (const candidate of candidates) {
    try {
      return JSON.parse(candidate);
    } catch (_) {
      // Try the next shape.
    }
  }
  return null;
}

function normalizeToolCall(call) {
  if (!call || typeof call !== "object") return null;
  const name = call.name || (call.function && call.function.name);
  if (!name) return null;
  const rawArguments = call.arguments !== undefined
    ? call.arguments
    : call.input !== undefined
      ? call.input
      : call.function && call.function.arguments !== undefined
        ? call.function.arguments
        : {};
  const args = typeof rawArguments === "string" ? parseJsonObject(rawArguments) : rawArguments;
  return {
    id: call.id || `call_${randomId()}`,
    name: String(name),
    arguments: args && typeof args === "object" && !Array.isArray(args) ? args : {},
  };
}

function parseJsonObject(value) {
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch (_) {
    return {};
  }
}

function toOpenAIChatToolCall(call) {
  return {
    id: call.id,
    type: "function",
    function: {
      name: call.name,
      arguments: JSON.stringify(call.arguments || {}),
    },
  };
}

function toOpenAIResponseToolCall(call) {
  return {
    id: `fc_${randomId()}`,
    type: "function_call",
    status: "completed",
    call_id: call.id,
    name: call.name,
    arguments: JSON.stringify(call.arguments || {}),
  };
}

function toAnthropicToolUse(call) {
  return {
    type: "tool_use",
    id: call.id,
    name: call.name,
    input: call.arguments || {},
  };
}

function openAIResponseEnvelope(body, data) {
  return {
    id: data.id,
    object: "response",
    created_at: data.created,
    status: data.status || "completed",
    error: null,
    incomplete_details: null,
    instructions: body.instructions || null,
    max_output_tokens: body.max_output_tokens || body.max_tokens || null,
    model: data.model,
    output: data.output || [],
    output_text: data.outputText || "",
    parallel_tool_calls: body.parallel_tool_calls !== false,
    previous_response_id: body.previous_response_id || null,
    reasoning: body.reasoning || null,
    store: body.store || false,
    temperature: body.temperature || null,
    text: body.text || { format: { type: "text" } },
    tool_choice: body.tool_choice || "auto",
    tools: body.tools || [],
    top_p: body.top_p || null,
    truncation: body.truncation || "disabled",
    usage: responseUsageFromText(data.usageInput || "", data.usageOutput || data.outputText || ""),
    user: body.user || null,
  };
}

async function openAIChatViaAnthropic(request, env, body, toolContext) {
  const anthropicBody = {
    model: toUpstreamAnthropicModel(body.model || env.DEFAULT_CLAUDE_MODEL || DEFAULT_CLAUDE_MODEL),
    max_tokens: body.max_tokens || body.max_completion_tokens || 4096,
    stream: false,
    system: collectOpenAISystemMessages(body.messages),
    messages: openAIChatMessagesToAnthropic(body.messages),
    tools: toolContext.tools.map(toAnthropicToolDefinition),
    tool_choice: toAnthropicToolChoice(body.tool_choice),
  };
  if (body.temperature != null) anthropicBody.temperature = body.temperature;
  if (body.top_p != null) anthropicBody.top_p = body.top_p;
  return callAnthropicMessagesJson(request, env, anthropicBody);
}

async function openAIResponsesViaAnthropic(request, env, body, toolContext) {
  const anthropicBody = {
    model: toUpstreamAnthropicModel(body.model || env.DEFAULT_CLAUDE_MODEL || DEFAULT_CLAUDE_MODEL),
    max_tokens: body.max_output_tokens || body.max_tokens || 4096,
    stream: false,
    system: body.instructions || "",
    messages: openAIResponseInputToAnthropic(body.input),
    tools: toolContext.tools.map(toAnthropicToolDefinition),
    tool_choice: toAnthropicToolChoice(body.tool_choice),
  };
  if (body.temperature != null) anthropicBody.temperature = body.temperature;
  if (body.top_p != null) anthropicBody.top_p = body.top_p;
  return callAnthropicMessagesJson(request, env, anthropicBody);
}

async function callAnthropicMessagesJson(request, env, body) {
  const response = await fetch(new URL("/v1/messages", upstreamBase(env)), {
    method: "POST",
    headers: anthropicUpstreamHeaders(request, env, false),
    body: JSON.stringify(body || {}),
  });

  if (!response.ok) {
    throw new Error(`upstream /v1/messages failed: ${response.status} ${await response.text()}`);
  }

  return response.json();
}

function openAIChatBridgeResponse(anthropic, meta) {
  const text = anthropicText(anthropic);
  const toolCalls = anthropicToolUses(anthropic).map((toolUse) => toOpenAIChatToolCall({
    id: toolUse.id,
    name: toolUse.name,
    arguments: toolUse.input || {},
  }));

  return {
    id: meta.id,
    object: "chat.completion",
    created: meta.created,
    model: meta.model,
    choices: [
      {
        index: 0,
        message: toolCalls.length
          ? { role: "assistant", content: text || null, tool_calls: toolCalls }
          : { role: "assistant", content: text },
        logprobs: null,
        finish_reason: toolCalls.length ? "tool_calls" : openAIStopReason(anthropic.stop_reason),
      },
    ],
    usage: openAIUsageFromAnthropic(anthropic.usage),
    system_fingerprint: "unlimited-surf-worker:anthropic-tool-bridge",
  };
}

function openAIResponsesBridgeResponse(body, anthropic, meta) {
  const text = anthropicText(anthropic);
  const toolOutputs = anthropicToolUses(anthropic).map((toolUse) => toOpenAIResponseToolCall({
    id: toolUse.id,
    name: toolUse.name,
    arguments: toolUse.input || {},
  }));
  const output = toolOutputs.length
    ? toolOutputs
    : [{
      id: `msg_${randomId()}`,
      type: "message",
      status: "completed",
      role: "assistant",
      content: [{ type: "output_text", text, annotations: [] }],
    }];

  return {
    ...openAIResponseEnvelope(body, {
      id: meta.id,
      created: meta.created,
      model: meta.model,
      output,
      outputText: toolOutputs.length ? "" : text,
      status: "completed",
      usageInput: "",
      usageOutput: text,
    }),
    usage: openAIResponseUsageFromAnthropic(anthropic.usage),
  };
}

function streamOpenAIChatBridge(anthropic, meta) {
  return new ReadableStream({
    start(controller) {
      const response = openAIChatBridgeResponse(anthropic, meta);
      const message = response.choices[0].message;
      writeSse(controller, {
        id: meta.id,
        object: "chat.completion.chunk",
        created: meta.created,
        model: meta.model,
        choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
      });
      if (message.tool_calls) {
        writeSse(controller, {
          id: meta.id,
          object: "chat.completion.chunk",
          created: meta.created,
          model: meta.model,
          choices: [{ index: 0, delta: { tool_calls: message.tool_calls }, finish_reason: null }],
        });
      } else if (message.content) {
        writeSse(controller, {
          id: meta.id,
          object: "chat.completion.chunk",
          created: meta.created,
          model: meta.model,
          choices: [{ index: 0, delta: { content: message.content }, finish_reason: null }],
        });
      }
      writeSse(controller, {
        id: meta.id,
        object: "chat.completion.chunk",
        created: meta.created,
        model: meta.model,
        choices: [{ index: 0, delta: {}, finish_reason: response.choices[0].finish_reason }],
      });
      writeRawSse(controller, "data: [DONE]\n\n");
      controller.close();
    },
  });
}

function streamOpenAIResponsesBridge(anthropic, meta) {
  return new ReadableStream({
    start(controller) {
      const outputId = `msg_${randomId()}`;
      const toolUses = anthropicToolUses(anthropic);
      const text = anthropicText(anthropic);
      writeSseEvent(controller, "response.created", {
        type: "response.created",
        response: { id: meta.id, object: "response", created_at: meta.created, status: "in_progress", model: meta.model, output: [] },
      });
      if (toolUses.length) {
        toolUses.forEach((toolUse, index) => {
          const item = toOpenAIResponseToolCall({ id: toolUse.id, name: toolUse.name, arguments: toolUse.input || {} });
          writeSseEvent(controller, "response.output_item.added", { type: "response.output_item.added", output_index: index, item });
          writeSseEvent(controller, "response.output_item.done", { type: "response.output_item.done", output_index: index, item });
        });
      } else {
        writeSseEvent(controller, "response.output_item.added", {
          type: "response.output_item.added",
          output_index: 0,
          item: { id: outputId, type: "message", status: "in_progress", role: "assistant", content: [] },
        });
        writeSseEvent(controller, "response.output_text.delta", {
          type: "response.output_text.delta",
          item_id: outputId,
          output_index: 0,
          content_index: 0,
          delta: text,
        });
      }
      writeSseEvent(controller, "response.completed", {
        type: "response.completed",
        response: { id: meta.id, object: "response", created_at: meta.created, status: "completed", model: meta.model },
      });
      writeRawSse(controller, "data: [DONE]\n\n");
      controller.close();
    },
  });
}

function responsesToChatBody(body, fallbackModel) {
  const messages = [];
  if (body.instructions) messages.push({ role: "system", content: body.instructions });
  const inputText = inputToText(body.input);
  if (inputText) messages.push({ role: "user", content: inputText });

  return {
    ...body,
    model: body.model || fallbackModel,
    messages,
    stream: body.stream,
  };
}

function collectOpenAISystemMessages(messages) {
  if (!Array.isArray(messages)) return "";
  return messages
    .filter((message) => message && (message.role === "system" || message.role === "developer"))
    .map((message) => contentToText(message.content))
    .filter(Boolean)
    .join("\n\n");
}

function openAIChatMessagesToAnthropic(messages) {
  if (!Array.isArray(messages) || !messages.length) return [{ role: "user", content: "" }];
  const converted = [];

  for (const message of messages) {
    if (!message || message.role === "system" || message.role === "developer") continue;

    if (message.role === "assistant") {
      const content = [];
      const text = contentToText(message.content);
      if (text) content.push({ type: "text", text });
      if (Array.isArray(message.tool_calls)) {
        for (const call of message.tool_calls) {
          const normalized = normalizeToolCall(call);
          if (normalized) content.push({ type: "tool_use", id: normalized.id, name: normalized.name, input: normalized.arguments || {} });
        }
      }
      converted.push({ role: "assistant", content: content.length ? content : [{ type: "text", text: "" }] });
      continue;
    }

    if (message.role === "tool") {
      converted.push({
        role: "user",
        content: [{
          type: "tool_result",
          tool_use_id: message.tool_call_id || message.id || `call_${randomId()}`,
          content: contentToText(message.content),
        }],
      });
      continue;
    }

    if (message.role === "function") {
      converted.push({
        role: "user",
        content: [{
          type: "tool_result",
          tool_use_id: message.tool_call_id || message.name || `call_${randomId()}`,
          content: contentToText(message.content),
        }],
      });
      continue;
    }

    converted.push({ role: "user", content: openAIContentToAnthropic(message.content) });
  }

  return mergeAdjacentAnthropicMessages(converted.length ? converted : [{ role: "user", content: "" }]);
}

function openAIResponseInputToAnthropic(input) {
  if (!input) return [{ role: "user", content: "" }];
  if (typeof input === "string") return [{ role: "user", content: input }];
  if (!Array.isArray(input)) return [{ role: "user", content: openAIContentToAnthropic(input) }];

  const messages = [];
  for (const item of input) {
    if (!item || typeof item !== "object") {
      messages.push({ role: "user", content: String(item || "") });
      continue;
    }

    if (item.type === "message" || item.role) {
      const role = item.role === "assistant" ? "assistant" : "user";
      messages.push({ role, content: openAIContentToAnthropic(item.content) });
      continue;
    }

    if (item.type === "function_call") {
      messages.push({
        role: "assistant",
        content: [{
          type: "tool_use",
          id: item.call_id || item.id || `call_${randomId()}`,
          name: item.name,
          input: parseJsonObject(item.arguments || "{}"),
        }],
      });
      continue;
    }

    if (item.type === "function_call_output") {
      messages.push({
        role: "user",
        content: [{
          type: "tool_result",
          tool_use_id: item.call_id || item.id || `call_${randomId()}`,
          content: contentToText(item.output || item.content || ""),
        }],
      });
      continue;
    }

    messages.push({ role: "user", content: openAIContentToAnthropic(item) });
  }

  return mergeAdjacentAnthropicMessages(messages.length ? messages : [{ role: "user", content: "" }]);
}

function openAIContentToAnthropic(content) {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return contentToText(content);

  const parts = [];
  for (const part of content) {
    if (!part || typeof part !== "object") {
      parts.push({ type: "text", text: String(part || "") });
    } else if ((part.type === "text" || part.type === "input_text" || part.type === "output_text") && typeof part.text === "string") {
      parts.push({ type: "text", text: part.text });
    } else if (part.type === "tool_result") {
      parts.push({
        type: "tool_result",
        tool_use_id: part.tool_use_id || part.call_id || `call_${randomId()}`,
        content: contentToText(part.content || part.output || ""),
      });
    } else {
      const text = contentToText(part);
      if (text) parts.push({ type: "text", text });
    }
  }
  return parts.length ? parts : "";
}

function mergeAdjacentAnthropicMessages(messages) {
  const merged = [];
  for (const message of messages) {
    const role = message.role === "assistant" ? "assistant" : "user";
    const content = message.content == null ? "" : message.content;
    const last = merged[merged.length - 1];
    if (last && last.role === role) {
      last.content = mergeAnthropicContent(last.content, content);
    } else {
      merged.push({ role, content });
    }
  }
  return merged;
}

function mergeAnthropicContent(left, right) {
  const leftParts = Array.isArray(left) ? left : left ? [{ type: "text", text: String(left) }] : [];
  const rightParts = Array.isArray(right) ? right : right ? [{ type: "text", text: String(right) }] : [];
  return [...leftParts, ...rightParts];
}

function toAnthropicToolDefinition(tool) {
  return {
    name: tool.name,
    description: tool.description || "",
    input_schema: tool.parameters || { type: "object", properties: {} },
  };
}

function toAnthropicToolChoice(choice) {
  if (!choice || choice === "auto") return { type: "auto" };
  if (choice === "none") return { type: "none" };
  if (choice === "required") return { type: "any" };
  if (typeof choice === "object") {
    const name = choice.name || (choice.function && choice.function.name);
    if (name) return { type: "tool", name };
    if (choice.type === "function" && choice.function && choice.function.name) return { type: "tool", name: choice.function.name };
  }
  return { type: "auto" };
}

function anthropicText(message) {
  if (!message || !Array.isArray(message.content)) return "";
  return message.content
    .filter((part) => part && part.type === "text" && typeof part.text === "string")
    .map((part) => part.text)
    .join("");
}

function anthropicToolUses(message) {
  if (!message || !Array.isArray(message.content)) return [];
  return message.content.filter((part) => part && part.type === "tool_use" && part.name);
}

function openAIUsageFromAnthropic(usage) {
  const promptTokens = Number(usage && usage.input_tokens) || 0;
  const completionTokens = Number(usage && usage.output_tokens) || 0;
  return { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens };
}

function openAIResponseUsageFromAnthropic(usage) {
  const inputTokens = Number(usage && usage.input_tokens) || 0;
  const outputTokens = Number(usage && usage.output_tokens) || 0;
  return { input_tokens: inputTokens, output_tokens: outputTokens, total_tokens: inputTokens + outputTokens };
}

function toUpstreamAnthropicModel(model) {
  const raw = String(model || DEFAULT_CLAUDE_MODEL);
  if (/^claude-.*-\d{8}$/.test(raw)) return raw;
  if (raw.startsWith("gateway-claude-opus-4-8")) return "claude-opus-4-8-20260501";
  if (raw.startsWith("gateway-claude-opus-4-7")) return "claude-opus-4-7-20260101";
  if (raw.startsWith("gateway-claude-opus-4-6")) return "claude-opus-4-6-20251201";
  if (raw.startsWith("gateway-claude-opus-4-5")) return "claude-opus-4-5-20251101";
  if (raw.startsWith("gateway-claude-opus-4-1")) return "claude-opus-4-1-20250805";
  if (raw.startsWith("gateway-claude-sonnet-4-6")) return "claude-sonnet-4-6-20260101";
  if (raw.startsWith("gateway-claude-sonnet-4")) return "claude-sonnet-4-20250514";
  if (/^claude-/i.test(raw)) return anthropicVersionedId(raw);
  return envDefaultClaudeModelSafe();
}

function envDefaultClaudeModelSafe() {
  return DEFAULT_CLAUDE_MODEL;
}

function normalizeAnthropicRequestBody(body, env) {
  const payload = { ...(body || {}) };
  payload.model = toUpstreamAnthropicModel(payload.model || (env && env.DEFAULT_CLAUDE_MODEL) || DEFAULT_CLAUDE_MODEL);
  return payload;
}

async function proxyUpstream(request, env, path) {
  const upstreamUrl = new URL(path + new URL(request.url).search, upstreamBase(env));
  const headers = new Headers(request.headers);
  const key = optionalUpstreamApiKey(request, env);
  if (key) headers.set("authorization", `Bearer ${key}`);
  headers.delete("host");

  const init = {
    method: request.method,
    headers,
    body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
    redirect: "manual",
  };

  const response = await fetch(upstreamUrl, init);
  return addCors(response);
}

async function callUnlimitedJson(request, env, path, payload) {
  const response = await fetch(new URL(path, upstreamBase(env)), {
    method: "POST",
    headers: upstreamHeaders(request, env, false),
    body: JSON.stringify(payload || {}),
  });

  if (!response.ok) {
    throw new Error(`upstream ${path} failed: ${response.status} ${await response.text()}`);
  }

  return response.json();
}

async function callUnlimitedStream(request, env, path, payload) {
  const response = await fetch(new URL(path, upstreamBase(env)), {
    method: "POST",
    headers: upstreamHeaders(request, env, true),
    body: JSON.stringify(payload || {}),
  });

  if (!response.ok) {
    throw new Error(`upstream ${path} failed: ${response.status} ${await response.text()}`);
  }

  return response;
}

async function collectUnlimitedText(request, env, path, payload) {
  const response = await callUnlimitedStream(request, env, path, payload);
  const events = await readUnlimitedEvents(response);
  let text = "";
  let finishReason = "stop";
  const annotations = [];

  for (const event of events) {
    if (typeof event.delta === "string") text += event.delta;
    if (event.results) annotations.push(event.results);
    if (event.finish && event.reason) finishReason = event.reason;
  }

  return { text, finishReason, annotations, rawEvents: events };
}

async function getModelCatalog(request, env) {
  try {
    const headers = new Headers();
    const key = optionalUpstreamApiKey(request, env);
    if (key) headers.set("Authorization", `Bearer ${key}`);
    const response = await fetch(new URL("/api/models", upstreamBase(env)), { headers });
    if (!response.ok) throw new Error(`models failed: ${response.status}`);
    const data = await response.json();
    const models = Array.isArray(data) ? data : Array.isArray(data.data) ? data.data : [];
    return models.map((model) => ({
      id: model.id || model.name || String(model),
      name: model.name || model.id || String(model),
      provider: model.provider || providerFromModel(model.id || model.name || ""),
      tier: model.tier || undefined,
    })).filter((model) => model.id);
  } catch (_) {
    return fallbackModels();
  }
}

function streamOpenAIChat(upstream, meta) {
  return streamUnlimitedEvents(upstream, {
    start(controller) {
      writeSse(controller, {
        id: meta.id,
        object: "chat.completion.chunk",
        created: meta.created,
        model: meta.model,
        choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
      });
    },
    delta(controller, text) {
      writeSse(controller, {
        id: meta.id,
        object: "chat.completion.chunk",
        created: meta.created,
        model: meta.model,
        choices: [{ index: 0, delta: { content: text }, finish_reason: null }],
      });
    },
    finish(controller, reason) {
      writeSse(controller, {
        id: meta.id,
        object: "chat.completion.chunk",
        created: meta.created,
        model: meta.model,
        choices: [{ index: 0, delta: {}, finish_reason: openAIStopReason(reason) }],
      });
      writeRawSse(controller, "data: [DONE]\n\n");
    },
  });
}

function streamOpenAIResponses(upstream, meta) {
  const outputId = `msg_${randomId()}`;
  return streamUnlimitedEvents(upstream, {
    start(controller) {
      writeSseEvent(controller, "response.created", {
        type: "response.created",
        response: {
          id: meta.id,
          object: "response",
          created_at: meta.created,
          status: "in_progress",
          model: meta.model,
          output: [],
        },
      });
      writeSseEvent(controller, "response.output_item.added", {
        type: "response.output_item.added",
        output_index: 0,
        item: { id: outputId, type: "message", status: "in_progress", role: "assistant", content: [] },
      });
      writeSseEvent(controller, "response.content_part.added", {
        type: "response.content_part.added",
        item_id: outputId,
        output_index: 0,
        content_index: 0,
        part: { type: "output_text", text: "", annotations: [] },
      });
    },
    delta(controller, text) {
      writeSseEvent(controller, "response.output_text.delta", {
        type: "response.output_text.delta",
        item_id: outputId,
        output_index: 0,
        content_index: 0,
        delta: text,
      });
    },
    finish(controller) {
      writeSseEvent(controller, "response.output_text.done", {
        type: "response.output_text.done",
        item_id: outputId,
        output_index: 0,
        content_index: 0,
        text: "",
      });
      writeSseEvent(controller, "response.content_part.done", {
        type: "response.content_part.done",
        item_id: outputId,
        output_index: 0,
        content_index: 0,
        part: { type: "output_text", text: "", annotations: [] },
      });
      writeSseEvent(controller, "response.output_item.done", {
        type: "response.output_item.done",
        output_index: 0,
        item: { id: outputId, type: "message", status: "completed", role: "assistant", content: [] },
      });
      writeSseEvent(controller, "response.completed", {
        type: "response.completed",
        response: { id: meta.id, object: "response", created_at: meta.created, status: "completed", model: meta.model },
      });
      writeRawSse(controller, "data: [DONE]\n\n");
    },
  });
}

function streamAnthropicMessages(upstream, meta) {
  return streamUnlimitedEvents(upstream, {
    start(controller) {
      writeSseEvent(controller, "message_start", {
        type: "message_start",
        message: {
          id: meta.id,
          type: "message",
          role: "assistant",
          model: meta.model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      });
      writeSseEvent(controller, "content_block_start", {
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "" },
      });
    },
    delta(controller, text) {
      writeSseEvent(controller, "content_block_delta", {
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text },
      });
    },
    finish(controller, reason) {
      writeSseEvent(controller, "content_block_stop", { type: "content_block_stop", index: 0 });
      writeSseEvent(controller, "message_delta", {
        type: "message_delta",
        delta: { stop_reason: anthropicStopReason(reason), stop_sequence: null },
        usage: { output_tokens: 0 },
      });
      writeSseEvent(controller, "message_stop", { type: "message_stop" });
    },
  });
}

function streamUnlimitedEvents(upstream, handlers) {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  return new ReadableStream({
    async start(controller) {
      let finished = false;
      handlers.start && handlers.start(controller);

      try {
        const reader = upstream.body.getReader();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split(/\r?\n/);
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const parsed = parseSseJson(line.slice(5).trim());
            if (!parsed) continue;

            if (typeof parsed.delta === "string" && parsed.delta.length) {
              handlers.delta && handlers.delta(controller, parsed.delta, parsed);
            }

            if (parsed.finish || parsed.done) {
              finished = true;
              handlers.finish && handlers.finish(controller, parsed.reason || "stop", parsed);
            }
          }
        }

        if (!finished) handlers.finish && handlers.finish(controller, "stop", {});
      } catch (error) {
        controller.enqueue(encoder.encode(`event: error\ndata: ${JSON.stringify({ error: error.message || String(error) })}\n\n`));
      } finally {
        controller.close();
      }
    },
  });
}

async function readUnlimitedEvents(response) {
  const decoder = new TextDecoder();
  const reader = response.body.getReader();
  const events = [];
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const parsed = parseSseJson(line.slice(5).trim());
      if (parsed) events.push(parsed);
    }
  }

  if (buffer.startsWith("data:")) {
    const parsed = parseSseJson(buffer.slice(5).trim());
    if (parsed) events.push(parsed);
  }

  return events;
}

function writeSse(controller, data) {
  writeRawSse(controller, `data: ${JSON.stringify(data)}\n\n`);
}

function writeSseEvent(controller, event, data) {
  writeRawSse(controller, `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

function writeRawSse(controller, chunk) {
  controller.enqueue(new TextEncoder().encode(chunk));
}

function sseResponse(body) {
  return new Response(body, {
    headers: {
      ...CORS_HEADERS,
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}

function jsonResponse(data, init = {}) {
  return new Response(JSON.stringify(data, null, 2), {
    ...init,
    headers: {
      ...CORS_HEADERS,
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      ...(init.headers || {}),
    },
  });
}

function textResponse(text, contentType, init = {}) {
  return new Response(text, {
    ...init,
    headers: {
      ...CORS_HEADERS,
      "Content-Type": contentType,
      "Cache-Control": "no-store",
      ...(init.headers || {}),
    },
  });
}

function errorResponse(status, code, message) {
  return jsonResponse({
    error: {
      message,
      type: code,
      code,
    },
  }, { status });
}

function addCors(response) {
  const headers = new Headers(response.headers);
  for (const [key, value] of Object.entries(CORS_HEADERS)) headers.set(key, value);
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

async function readJson(request) {
  if (!request.body) return {};
  const text = await request.text();
  if (!text.trim()) return {};
  try {
    return JSON.parse(text);
  } catch (_) {
    throw new Error("Request body must be valid JSON.");
  }
}

function upstreamHeaders(request, env, wantsStream) {
  const headers = new Headers();
  headers.set("Authorization", `Bearer ${upstreamApiKey(request, env)}`);
  headers.set("Content-Type", "application/json");
  if (wantsStream) headers.set("Accept", "text/event-stream");
  return headers;
}

function anthropicUpstreamHeaders(request, env, wantsStream) {
  const key = upstreamApiKey(request, env);
  const headers = new Headers();
  headers.set("Content-Type", "application/json");
  headers.set("x-api-key", key);
  headers.set("anthropic-api-key", key);
  headers.set("Authorization", `Bearer ${key}`);
  headers.set("anthropic-version", request.headers.get("anthropic-version") || "2023-06-01");
  const beta = request.headers.get("anthropic-beta");
  if (beta) headers.set("anthropic-beta", beta);
  if (wantsStream) headers.set("Accept", "text/event-stream");
  return headers;
}

function upstreamApiKey(request, env) {
  const key = optionalUpstreamApiKey(request, env);
  if (key) return key;

  if (env.WORKER_API_KEY) {
    throw new Error("Missing upstream API key. Set UNLIMITED_SURF_API_KEY when WORKER_API_KEY is enabled.");
  }

  throw new Error("Missing upstream API key. Set UNLIMITED_SURF_API_KEY or pass Authorization: Bearer <key> / x-api-key: <key>.");
}

function optionalUpstreamApiKey(request, env) {
  const configured = env.UNLIMITED_SURF_API_KEY || env.API_KEY || env.AUTH_KEY;
  if (configured) return configured;

  if (env.WORKER_API_KEY) return "";

  return clientApiKey(request);
}

function validateWorkerApiKey(request, env) {
  const expected = env.WORKER_API_KEY;
  if (!expected) return null;

  const actual = clientApiKey(request);
  if (actual && constantTimeEqual(actual, expected)) return null;

  return jsonResponse({
    error: {
      message: "Invalid or missing Worker API key.",
      type: "authentication_error",
      code: "invalid_api_key",
    },
  }, { status: 401, headers: { "WWW-Authenticate": "Bearer" } });
}

function clientApiKey(request) {
  const auth = request.headers.get("authorization") || "";
  if (/^bearer\s+/i.test(auth)) return auth.replace(/^bearer\s+/i, "").trim();

  const xKey = request.headers.get("x-api-key") || request.headers.get("anthropic-api-key");
  return xKey ? xKey.trim() : "";
}

function constantTimeEqual(actual, expected) {
  const actualText = String(actual || "");
  const expectedText = String(expected || "");
  if (actualText.length !== expectedText.length) return false;

  let diff = 0;
  for (let i = 0; i < actualText.length; i += 1) {
    diff |= actualText.charCodeAt(i) ^ expectedText.charCodeAt(i);
  }
  return diff === 0;
}

function upstreamBase(env) {
  return stripTrailingSlash(env.UPSTREAM_BASE_URL || DEFAULT_UPSTREAM_BASE_URL) + "/";
}

function normalizePath(path) {
  if (!path || path === "") return "/";
  const normalized = path.replace(/\/+/g, "/");
  return normalized.length > 1 ? normalized.replace(/\/+$/, "") : normalized;
}

function messagesToText(messages) {
  if (!Array.isArray(messages)) return "";
  return messages.map((message) => {
    const role = message.role || "user";
    return `${role}: ${contentToText(message.content)}`;
  }).filter(Boolean).join("\n\n");
}

function anthropicMessagesToText(body) {
  const parts = [];
  if (body.system) parts.push(`system: ${contentToText(body.system)}`);
  if (Array.isArray(body.tools) && body.tools.length) {
    parts.push(`available tools: ${JSON.stringify(body.tools)}`);
    parts.push("If a tool is required, describe the intended tool call clearly. MCP and local tools must be executed by the client agent.");
  }
  if (Array.isArray(body.messages)) parts.push(messagesToText(body.messages));
  return parts.filter(Boolean).join("\n\n");
}

function inputToText(input) {
  if (!input) return "";
  if (typeof input === "string") return input;
  if (!Array.isArray(input)) return contentToText(input);

  return input.map((item) => {
    if (typeof item === "string") return item;
    if (item.type === "message") return `${item.role || "user"}: ${contentToText(item.content)}`;
    if (item.role) return `${item.role}: ${contentToText(item.content)}`;
    if (item.type === "input_text" || item.type === "output_text") return item.text || "";
    return contentToText(item);
  }).filter(Boolean).join("\n\n");
}

function contentToText(content) {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map((part) => contentToText(part)).filter(Boolean).join("\n");
  }
  if (typeof content === "object") {
    if (typeof content.text === "string") return content.text;
    if (typeof content.input_text === "string") return content.input_text;
    if (content.type === "text" && typeof content.text === "string") return content.text;
    if (content.type === "input_text" && typeof content.text === "string") return content.text;
    if (content.type === "image_url") return `[image: ${content.image_url && content.image_url.url ? content.image_url.url : "attached"}]`;
    if (content.type === "image") return "[image attached]";
    if (content.type === "tool_result") return `[tool_result ${content.tool_use_id || ""}] ${contentToText(content.content)}`;
    if (content.type === "tool_use") return `[tool_use ${content.name || "tool"}] ${JSON.stringify(content.input || {})}`;
    if (content.type) return `[${content.type}] ${JSON.stringify(content)}`;
  }
  return String(content);
}

function latestUserText(messages) {
  if (!Array.isArray(messages)) return "";
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if ((messages[i].role || "user") === "user") return contentToText(messages[i].content);
  }
  return "";
}

function hasWebSearchTool(tools) {
  if (!Array.isArray(tools)) return false;
  return tools.some((tool) => {
    const type = tool && (tool.type || tool.name || (tool.function && tool.function.name));
    return /web.?search|browser|search/i.test(String(type || ""));
  });
}

function reasoningEffort(body) {
  if (body.effort) return body.effort;
  if (typeof body.reasoning_effort === "string") return body.reasoning_effort;
  if (body.reasoning && typeof body.reasoning.effort === "string") return body.reasoning.effort;
  return "medium";
}

function mapUpstreamModel(model) {
  if (!model) return DEFAULT_OPENAI_MODEL;
  if (model.startsWith("gateway-")) return model;
  if (/^claude-/i.test(model)) return `gateway-${model.replace(/-\d{8}$/, "")}`;
  if (/^gpt-/i.test(model)) return `gateway-${model}`;
  if (/^gemini-/i.test(model)) return `gateway-google-${model.replace(/^gemini-/i, "")}`;
  return model;
}

function toAnthropicModel(model) {
  const id = model.id.startsWith("gateway-") ? model.id.replace(/^gateway-/, "") : model.id;
  const versioned = /^claude-.*-\d{8}$/.test(id) ? id : anthropicVersionedId(id);
  return {
    id: versioned,
    type: "model",
    display_name: model.name || versioned,
    created_at: "2026-01-01T00:00:00Z",
  };
}

function anthropicVersionedId(id) {
  if (/^claude-opus-4-8$/i.test(id)) return "claude-opus-4-8-20260501";
  if (/^claude-opus-4-7$/i.test(id)) return "claude-opus-4-7-20260101";
  if (/^claude-opus-4-6$/i.test(id)) return "claude-opus-4-6-20251201";
  if (/^claude-opus-4-5$/i.test(id)) return "claude-opus-4-5-20251101";
  if (/^claude-opus-4-1$/i.test(id)) return "claude-opus-4-1-20250805";
  if (/^claude-sonnet-4-6$/i.test(id)) return "claude-sonnet-4-6-20260101";
  if (/^claude-sonnet-4$/i.test(id)) return "claude-sonnet-4-20250514";
  if (/^claude-/i.test(id)) return `${id}-20260101`;
  return id;
}

function providerFromModel(model) {
  if (/claude|anthropic/i.test(model)) return "anthropic";
  if (/gemini|google/i.test(model)) return "google";
  if (/gpt|openai/i.test(model)) return "openai";
  return "unlimited.surf";
}

function fallbackModels() {
  return [
    { id: "gateway-gpt-5", name: "GPT-5", provider: "openai", tier: "flagship" },
    { id: "gateway-gpt-5-1", name: "GPT-5.1", provider: "openai", tier: "flagship" },
    { id: "gateway-claude-opus-4-7", name: "Claude Opus 4.7", provider: "anthropic", tier: "flagship" },
    { id: "gateway-google-2.5-pro", name: "Gemini 2.5 Pro", provider: "google", tier: "flagship" },
    { id: "gateway-gemini-3-flash", name: "Gemini 3 Flash", provider: "google", tier: "fast" },
  ];
}

function parseSseJson(data) {
  if (!data || data === "[DONE]") return null;
  try {
    return JSON.parse(data);
  } catch (_) {
    return null;
  }
}

function openAIStopReason(reason) {
  if (!reason) return "stop";
  if (reason === "max_tokens") return "length";
  if (reason === "tool_use") return "tool_calls";
  return reason === "end_turn" ? "stop" : reason;
}

function anthropicStopReason(reason) {
  if (!reason || reason === "stop") return "end_turn";
  if (reason === "length") return "max_tokens";
  if (reason === "tool_calls") return "tool_use";
  return reason;
}

function usageFromText(input, output) {
  const promptTokens = estimateTokens(input);
  const completionTokens = estimateTokens(output);
  return { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens };
}

function responseUsageFromText(input, output) {
  const inputTokens = estimateTokens(input);
  const outputTokens = estimateTokens(output);
  return { input_tokens: inputTokens, output_tokens: outputTokens, total_tokens: inputTokens + outputTokens };
}

function anthropicUsageFromText(input, output) {
  return { input_tokens: estimateTokens(input), output_tokens: estimateTokens(output) };
}

function estimateTokens(text) {
  if (!text) return 0;
  return Math.max(1, Math.ceil(String(text).length / 4));
}

function nowSeconds() {
  return Math.floor(Date.now() / 1000);
}

function randomId() {
  const bytes = new Uint8Array(12);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function stripTrailingSlash(value) {
  return String(value || "").replace(/\/+$/, "");
}

function bytesToBase64(bytes) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

function looksLikeAnthropicRequest(request) {
  return request.headers.has("anthropic-version") || request.headers.has("anthropic-beta") || request.headers.has("x-api-key");
}

function serviceInfo(request, env) {
  const origin = new URL(request.url).origin;
  return {
    ok: true,
    service: "unlimited.surf OpenAI/Anthropic compatibility Worker",
    upstream: stripTrailingSlash(env.UPSTREAM_BASE_URL || DEFAULT_UPSTREAM_BASE_URL),
    routes: {
      raw: `${origin}/api/chat, /api/search, /api/merge, /api/models, /api/key, /api/attachments/extract`,
      openai: `${origin}/v1/chat/completions, /v1/responses, /v1/models, /v1/files`,
      anthropic: `${origin}/v1/messages or ${origin}/anthropic/v1/messages`,
      setup: `${origin}/v1/setup, /v1/codex, /v1/mcp`,
    },
  };
}

function agentSetup(request, env) {
  const origin = new URL(request.url).origin;
  const claudeModel = toUpstreamAnthropicModel(env && env.DEFAULT_CLAUDE_MODEL ? env.DEFAULT_CLAUDE_MODEL : DEFAULT_CLAUDE_MODEL);
  return `Claude Code / Anthropic-compatible setup

PowerShell:
$env:ANTHROPIC_BASE_URL = "${origin}"
$env:ANTHROPIC_AUTH_TOKEN = "<your unlimited.surf key>"
$env:ANTHROPIC_API_KEY = "<your unlimited.surf key>"
$env:ANTHROPIC_MODEL = "${claudeModel}"
claude

Bash:
export ANTHROPIC_BASE_URL="${origin}"
export ANTHROPIC_AUTH_TOKEN="<your unlimited.surf key>"
export ANTHROPIC_API_KEY="<your unlimited.surf key>"
export ANTHROPIC_MODEL="${claudeModel}"
claude

Goose / Hermes / other agents:
Provider: Anthropic-compatible
Base URL: ${origin}
API key: <your unlimited.surf key>
Model: ${claudeModel}

Messages endpoint: POST ${origin}/v1/messages
Models endpoint: GET ${origin}/v1/models

MCP tools run in the client/agent environment. Use this Worker as the model endpoint, then configure MCP servers in your IDE or agent.
`;
}

function codexSetup(request) {
  const origin = new URL(request.url).origin;
  return `Codex custom provider notes

OpenAI-compatible Chat Completions:
base_url = "${origin}/v1"
api_key = "<your unlimited.surf key>"
model = "${DEFAULT_OPENAI_MODEL}"

OpenAI Responses-compatible route for newer agents:
POST ${origin}/v1/responses

Direct smoke test:
curl ${origin}/v1/chat/completions \\
  -H "Authorization: Bearer <your unlimited.surf key>" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"${DEFAULT_OPENAI_MODEL}","messages":[{"role":"user","content":"Write a small test function."}],"stream":true}'

Anthropic-compatible agent route:
POST ${origin}/v1/messages

MCP execution remains client-side; configure MCP servers in Codex or your IDE, and point the model provider at this Worker.
`;
}

function mcpInfo(request) {
  const origin = new URL(request.url).origin;
  return {
    supported: true,
    model_endpoint: origin,
    note: "MCP servers execute inside the client or agent. This Worker supplies OpenAI/Anthropic-compatible model endpoints and does not run local MCP tools in the browser or edge runtime.",
    endpoints: {
      openai_responses: `${origin}/v1/responses`,
      openai_chat_completions: `${origin}/v1/chat/completions`,
      anthropic_messages: `${origin}/v1/messages`,
      setup: `${origin}/v1/setup`,
    },
  };
}
