/**
 * Pi adapter. Run: node pi.mjs CONFIG_JSON_PATH
 *
 * The parent owns credentials, billing, and all cloud effects. Only explicitly
 * supplied bridge tools are available; Pi does not discover local resources.
 */
import { appendFileSync, mkdirSync, readFileSync } from "node:fs";
import { resolve, join } from "node:path";
import { pathToFileURL } from "node:url";
import { requestJSON, RUNTIME_REQUEST_TIMEOUT_MS } from "./local_http.mjs";
import { request as httpRequest } from "node:http";

function validateConfig(config) {
  for (const field of ["model", "system", "prompt", "bridge_url", "bridge_token", "session_dir"]) {
    if (typeof config[field] !== "string" || !config[field].trim()) {
      throw new Error(`Pi config requires a nonempty ${field}`);
    }
  }
  const bridge = new URL(config.bridge_url);
  if (!["http:", "https:"].includes(bridge.protocol) || bridge.username || bridge.password ||
      bridge.search || bridge.hash || !["", "/"].includes(bridge.pathname)) {
    throw new Error("Pi bridge_url must be an HTTP(S) origin without credentials or a path");
  }
  if (!Number.isSafeInteger(config.max_turns) || config.max_turns < 1) {
    throw new Error("Pi max_turns must be a positive integer");
  }
  const maxTokens = config.max_tokens ?? 16000;
  if (!Number.isSafeInteger(maxTokens) || maxTokens < 1) {
    throw new Error("Pi max_tokens must be a positive integer");
  }
  if (!Number.isFinite(config.model_timeout_sec ?? 300) || (config.model_timeout_sec ?? 300) <= 0) {
    throw new Error("Pi model_timeout_sec must be positive");
  }
  if (!Array.isArray(config.tools)) throw new Error("Pi tools must be an array");
  const names = new Set();
  for (const tool of config.tools) {
    const fn = tool?.function;
    if (tool?.type !== "function" || typeof fn?.name !== "string" ||
        !/^[a-zA-Z0-9_-]{1,64}$/.test(fn.name) || names.has(fn.name) ||
        fn.parameters?.type !== "object") {
      throw new Error("Pi tools require unique OpenAI function schemas with object parameters");
    }
    names.add(fn.name);
  }
  return { bridge, maxTokens };
}

function textContent(content) {
  if (typeof content === "string") return content;
  return (content ?? []).filter((block) => block.type === "text").map((block) => block.text).join("\n");
}

function normalizedMessages(system, messages) {
  const output = [{ role: "system", content: system }];
  for (const message of messages) {
    if (message.role === "user") {
      output.push({ role: "user", content: textContent(message.content) });
    } else if (message.role === "assistant") {
      const toolCalls = message.content.filter((block) => block.type === "toolCall").map((block) => ({
        id: block.id,
        type: "function",
        function: { name: block.name, arguments: JSON.stringify(block.arguments) },
      }));
      output.push({
        role: "assistant",
        content: textContent(message.content),
        ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
      });
    } else if (message.role === "toolResult") {
      output.push({
        role: "tool", tool_call_id: message.toolCallId, name: message.toolName,
        content: textContent(message.content),
      });
    }
  }
  return output;
}

function emptyResourceLoader(createExtensionRuntime, system) {
  const extensions = { extensions: [], errors: [], runtime: createExtensionRuntime() };
  return {
    getExtensions: () => extensions,
    getSkills: () => ({ skills: [], diagnostics: [] }),
    getPrompts: () => ({ prompts: [], diagnostics: [] }),
    getThemes: () => ({ themes: [], diagnostics: [] }),
    getAgentsFiles: () => ({ agentsFiles: [] }),
    getSystemPrompt: () => system,
    getSystemPromptSource: () => undefined,
    getAppendSystemPrompt: () => [],
    getAppendSystemPromptSources: () => [],
    extendResources: () => {},
    reload: async () => {},
  };
}

export async function runPi(config) {
  const { bridge, maxTokens } = validateConfig(config);
  const inference = config.inference_options ?? {};
  const adaptive = inference.thinking?.type === "adaptive";
  const requestTimeoutMs = (config.model_timeout_sec ?? 300) * 1000;
  const sessionDir = resolve(config.session_dir);
  mkdirSync(sessionDir, { recursive: true });
  const eventsPath = join(sessionDir, "pi-events.jsonl");
  const redact = (value) => JSON.stringify(value, (_key, item) => {
    if (item instanceof Error) return { name: item.name, message: item.message };
    return item;
  }).split(config.bridge_token).join("[bridge-token]");
  const log = (event) => appendFileSync(eventsPath, `${redact(event)}\n`, { mode: 0o600 });
  const controller = new AbortController();
  let session;
  let turns = 0;
  let interrupted;
  const onSignal = (signal) => {
    interrupted = signal;
    controller.abort(new Error(`Pi interrupted by ${signal}`));
    if (session) void session.abort();
  };
  const onSigint = () => onSignal("SIGINT");
  const onSigterm = () => onSignal("SIGTERM");
  process.on("SIGINT", onSigint);
  process.on("SIGTERM", onSigterm);
  const originalOffline = process.env.PI_OFFLINE;
  process.env.PI_OFFLINE = "1";

  // This also refuses redirects, so the local bridge token never follows a
  // redirect to a provider or a URL selected by model output.
  //
  // node:http honours only the explicit deadline below. Native fetch applies an
  // implicit 300s headers/idle-body timeout that no AbortSignal can raise, and
  // a silent gateway would kill a live model request mid-stall-recovery at
  // exactly t+300s — before the bridge's own attempt gates can settle it (the
  // trap local_http.mjs documents for control requests).
  const bridgeFetch = async (input, init = {}) => {
    const url = new URL(input instanceof Request ? input.url : input);
    if (url.origin !== bridge.origin || !["/v1/messages", "/tool"].includes(url.pathname)) {
      throw new Error("Pi attempted an HTTP request outside its configured bridge");
    }
    const signals = [controller.signal, AbortSignal.timeout(requestTimeoutMs)];
    if (init.signal) signals.push(init.signal);
    if (input instanceof Request && input.signal) signals.push(input.signal);
    // Spread would drop a Headers instance's pairs (they are not own
    // enumerable props) and lose the bridge token, so normalize first.
    const source = input instanceof Request ? input.headers : init.headers;
    const headers = source ? Object.fromEntries(new Headers(source)) : {};
    // Never request an encoding this hand-built client would receive raw.
    delete headers["accept-encoding"];
    const method = init.method ?? (input instanceof Request ? input.method : undefined) ?? "GET";
    let body = init.body ?? (input instanceof Request && !["GET", "HEAD"].includes(input.method) ? input.body : undefined);
    if (body !== undefined && typeof body === "object" && !Buffer.isBuffer(body) && !(body instanceof Uint8Array)) {
      body = Buffer.from(await new Response(body).arrayBuffer());
    }
    const payload = body === undefined ? undefined : Buffer.isBuffer(body) ? body : Buffer.from(body);
    return new Promise((resolve, reject) => {
      let settled = false;
      let request;
      try {
        request = httpRequest(
          url,
          {
            method,
            headers: { ...headers, ...(payload === undefined ? {} : { "Content-Length": Buffer.byteLength(payload) }) },
            signal: AbortSignal.any(signals),
            agent: false,
          },
          (response) => {
            const status = response.statusCode ?? 599;
            if (status >= 300 && status < 400) {
              response.resume();
              reject(new Error(`Pi bridge refused redirect to HTTP ${status}`));
              return;
            }
            settled = true;
            const stream = new ReadableStream({
              start(target) {
                let ended = false;
                response.on("data", (chunk) => target.enqueue(new Uint8Array(chunk)));
                response.once("end", () => {
                  ended = true;
                  target.close();
                });
                response.once("error", (error) => target.error(error));
                response.once("aborted", () => target.error(new Error("Pi bridge response aborted before completion")));
                response.once("close", () => {
                  if (!ended) target.error(new Error("Pi bridge response closed before completion"));
                });
              },
              cancel() {
                response.destroy();
                request.destroy();
              },
            });
            const responseHeaders = new Headers();
            for (const [name, value] of Object.entries(response.headers)) {
              responseHeaders.set(name, Array.isArray(value) ? value.join(", ") : String(value));
            }
            resolve(new Response(stream, { status, statusText: response.statusMessage ?? "", headers: responseHeaders }));
          },
        );
      } catch (error) {
        reject(new Error(`Pi bridge request ${url.pathname} failed: ${error.message}`));
        return;
      }
      request.once("error", (error) => {
        if (!settled) reject(new Error(`Pi bridge transport ${url.pathname}: ${error.message}`));
      });
      request.end(payload);
    });
  };

  try {
    let sdk;
    let ai;
    try {
      sdk = await import("@earendil-works/pi-coding-agent");
      ai = await import("@earendil-works/pi-ai");
    } catch (error) {
      throw new Error(
        `Pi SDK import failed; run repo2rlenv tasksmith install-runtime to install the pinned SDKs: ${error.message}`,
        { cause: error },
      );
    }
    const { createAgentSession, createExtensionRuntime, ModelRuntime, SessionManager, SettingsManager } = sdk;
    const credentials = new ai.InMemoryCredentialStore();
    await credentials.modify("anthropic", async () => ({ type: "api_key", key: config.bridge_token }));
    const modelRuntime = await ModelRuntime.create({
      credentials, modelsPath: null, allowModelNetwork: false, refreshOnCreate: false,
      signal: controller.signal,
    });
    const settingsManager = SettingsManager.inMemory({
      compaction: { enabled: false },
      retry: { enabled: false, maxRetries: 0, provider: { maxRetries: 0 } },
      enableAnalytics: false,
      enableInstallTelemetry: false,
      enableSkillCommands: false,
      transport: "sse",
    });
    const customTools = config.tools.map(({ function: fn }) => ({
      name: fn.name,
      label: fn.name,
      description: fn.description ?? "",
      parameters: fn.parameters,
      execute: async (_id, args, signal) => {
        const result = await requestJSON(new URL("/tool", bridge), {
          headers: { Authorization: `Bearer ${config.bridge_token}`, "Content-Type": "application/json" },
          body: { name: fn.name, arguments: args },
          signal: signal ? AbortSignal.any([signal, controller.signal]) : controller.signal,
          timeoutMs: RUNTIME_REQUEST_TIMEOUT_MS,
          label: `Pi tool bridge ${fn.name}`,
        });
        if (typeof result.output !== "string") throw new Error(`Pi tool bridge ${fn.name} returned no string output`);
        return { content: [{ type: "text", text: result.output }], details: {} };
      },
    }));
    const model = {
      id: config.model,
      name: config.model,
      provider: "anthropic",
      api: "anthropic-messages",
      baseUrl: bridge.origin,
      reasoning: adaptive,
      input: ["text"],
      // Parent computes actual usage/cost and rejects requests over its budget.
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 200000,
      maxTokens,
    };
    ({ session } = await createAgentSession({
      cwd: sessionDir,
      agentDir: sessionDir,
      model,
      modelRuntime,
      thinkingLevel: adaptive ? "medium" : "off",
      noTools: "builtin",
      tools: customTools.map((tool) => tool.name),
      customTools,
      resourceLoader: emptyResourceLoader(createExtensionRuntime, config.system),
      settingsManager,
      sessionManager: SessionManager.create(sessionDir, sessionDir),
    }));
    const activeTools = session.getActiveToolNames();
    if (activeTools.length !== customTools.length ||
        customTools.some((tool) => !activeTools.includes(tool.name) || session.getToolDefinition(tool.name) !== tool)) {
      throw new Error("Pi active tools differ from the supplied bridge tools");
    }
    const piStream = session.agent.streamFunction;
    session.agent.streamFunction = (currentModel, context, options) => {
      if (turns >= config.max_turns) throw new Error("Pi exceeded max_turns");
      turns += 1;
      return piStream(currentModel, context, {
        ...options, maxTokens, maxRetries: 0, fetch: bridgeFetch,
        // Pi resolves each request's model timeout as
        // `options.timeoutMs ?? retry.provider.timeoutMs ?? httpIdleTimeoutMs`,
        // and its HTTP-idle default is 5 minutes. That default capped every
        // model stream at t+300s, killing pi mid stall-recovery (240+560) even
        // though model_timeout_sec requires the runtime's own abort to stay
        // above that window. Pass the matched deadline so it, not the idle
        // setting, governs the upstream stack.
        timeoutMs: requestTimeoutMs,
        // The parent selects the same policy for every runtime. Override Pi's
        // model-catalog defaults without touching signed conversation blocks.
        onPayload: (payload) => {
          // Pi 1.x carries system messages in the transcript and appends local
          // context. Keep the exact shared prompt at the provider boundary.
          payload.system = [{ type: "text", text: config.system }];
          for (const key of ["thinking", "temperature", "top_p", "top_k", "output_config"]) delete payload[key];
          Object.assign(payload, inference, { max_tokens: maxTokens });
          // Pi's non-strict Anthropic converter retains only type,
          // properties and required. That leaves Pydantic's nested $refs
          // unresolved and hides root constraints from the model, although
          // Pi's local validator still enforces them. Preserve the supplied
          // schema without enabling strict sampling or rewriting its meaning.
          for (const tool of payload.tools ?? []) {
            const definition = config.tools.find(({ function: fn }) => fn.name === tool.name);
            if (!definition) throw new Error(`Pi emitted an unregistered tool: ${tool.name}`);
            tool.input_schema = structuredClone(definition.function.parameters);
          }
          return payload;
        },
      });
    };
    const finishTurn = session.agent.finishTurn;
    session.agent.finishTurn = async (...args) => {
      const decision = await finishTurn?.call(session.agent, ...args);
      // Pi 1.x finishes this hook after all tool results, so the last permitted
      // validation callback completes before the budget ends the run.
      return turns >= config.max_turns ? { action: "end" } : decision;
    };
    session.subscribe(log);
    log({ type: "adapter_start", runtime: "pi", model: config.model, tools: activeTools, max_turns: config.max_turns, max_tokens: maxTokens, inference_options: inference, model_timeout_sec: requestTimeoutMs / 1000 });
    controller.signal.throwIfAborted();
    await session.prompt(config.prompt);
    if (interrupted) throw new Error(`Pi interrupted by ${interrupted}`);
    const lastAssistant = session.messages.findLast((message) => message.role === "assistant");
    if (lastAssistant?.stopReason === "error" || lastAssistant?.stopReason === "aborted") {
      throw new Error(`Pi provider failed: ${lastAssistant.errorMessage || lastAssistant.stopReason}`);
    }
    if (["length", "max_tokens"].includes(lastAssistant?.stopReason)) {
      throw new Error(`Pi provider response truncated: ${lastAssistant.stopReason}`);
    }
    if (!lastAssistant || (!textContent(lastAssistant.content).trim() &&
        !(lastAssistant.content ?? []).some((block) => block.type === "toolCall"))) {
      throw new Error("Pi provider response contained no text or tool calls (thinking-only or empty)");
    }
    const result = { messages: normalizedMessages(config.system, session.messages), turns, cost: 0 };
    log({ type: "adapter_end", turns, session_file: session.sessionFile });
    return JSON.parse(redact(result));
  } catch (error) {
    const result = {
      messages: normalizedMessages(config.system, session?.messages ?? []),
      turns, cost: 0, error: error instanceof Error ? error.message : String(error),
      ...(interrupted ? { interrupted } : {}),
    };
    log({ type: "adapter_error", error: result.error, interrupted, turns });
    return JSON.parse(redact(result));
  } finally {
    session?.dispose();
    process.off("SIGINT", onSigint);
    process.off("SIGTERM", onSigterm);
    if (originalOffline === undefined) delete process.env.PI_OFFLINE;
    else process.env.PI_OFFLINE = originalOffline;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    if (process.argv.length !== 3) throw new Error("Usage: node pi.mjs CONFIG_JSON_PATH");
    const result = await runPi(JSON.parse(readFileSync(process.argv[2], "utf8")));
    process.stdout.write(`${JSON.stringify(result)}\n`);
    process.exitCode = result.interrupted === "SIGINT" ? 130 : result.interrupted === "SIGTERM" ? 143 : result.error ? 1 : 0;
  } catch (error) {
    process.stdout.write(`${JSON.stringify({ messages: [], turns: 0, cost: 0, error: error instanceof Error ? error.message : String(error) })}\n`);
    process.exitCode = 1;
  }
}
