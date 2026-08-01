const DEFAULT_TIMEOUT_MS = 30_000;
const MAX_ERROR_BODY_LENGTH = 4_096;

export const API_TIMEOUTS = {
  default: DEFAULT_TIMEOUT_MS,
  download: 120_000,
  upload: 30 * 60_000,
  stream: 2 * 60 * 60_000,
} as const;

export class ApiError extends Error {
  readonly status: number;
  readonly details?: unknown;

  constructor(message: string, status: number, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

export class ApiTimeoutError extends Error {
  readonly timeoutMs: number;

  constructor(timeoutMs: number) {
    super(`API request timed out after ${Math.ceil(timeoutMs / 1_000)} seconds`);
    this.name = "ApiTimeoutError";
    this.timeoutMs = timeoutMs;
  }
}

export interface ApiRequestOptions extends Omit<RequestInit, "body"> {
  body?: BodyInit | null;
  json?: unknown;
  timeoutMs?: number;
}

export interface ApiRequestHandle {
  cancel: (reason?: string) => void;
}

interface RequestScope {
  signal: AbortSignal;
  close: () => void;
}

interface SseHandlers<T> {
  onMessage: (data: T) => void;
  onError: (error: unknown) => void;
  onComplete: () => void;
}

export interface ParsedSseEvent {
  data: string;
  event?: string;
  id?: string;
}

function assertApiPath(path: string): string {
  if (path !== "/api" && !path.startsWith("/api/")) {
    throw new TypeError("API requests must use a same-origin /api path");
  }
  return path;
}

/**
 * Build a URL that can safely be used by fetch, links, iframes, and downloads.
 * This deliberately has no environment-controlled or absolute URL escape hatch.
 */
export function apiUrl(path: string): string {
  return assertApiPath(path);
}

export function apiPathSegment(value: string): string {
  if (value === "." || value === "..") {
    throw new TypeError("Dot path segments are not valid API resource identifiers");
  }
  return encodeURIComponent(value);
}

function createRequestScope(externalSignal: AbortSignal | null | undefined, timeoutMs: number): RequestScope {
  const controller = new AbortController();
  let timeoutId: ReturnType<typeof setTimeout> | undefined;

  const abortFromExternal = () => {
    controller.abort(externalSignal?.reason);
  };

  if (externalSignal?.aborted) {
    abortFromExternal();
  } else if (externalSignal) {
    externalSignal.addEventListener("abort", abortFromExternal, { once: true });
  }

  if (timeoutMs > 0) {
    timeoutId = setTimeout(() => {
      controller.abort(new ApiTimeoutError(timeoutMs));
    }, timeoutMs);
  }

  return {
    signal: controller.signal,
    close: () => {
      if (timeoutId !== undefined) clearTimeout(timeoutId);
      externalSignal?.removeEventListener("abort", abortFromExternal);
    },
  };
}

function errorMessageValue(payload: unknown): string | null {
  if (typeof payload === "string") return payload.trim() || null;
  if (!payload || typeof payload !== "object") return null;

  const record = payload as Record<string, unknown>;
  const candidate = record.detail ?? record.message ?? record.error;
  if (typeof candidate === "string") return candidate.trim() || null;
  if (candidate && typeof candidate === "object") {
    const nested = candidate as Record<string, unknown>;
    if (typeof nested.message === "string") return nested.message.trim() || null;
  }
  if (candidate !== undefined) {
    try {
      return JSON.stringify(candidate);
    } catch {
      return String(candidate);
    }
  }
  return null;
}

async function responseError(response: Response): Promise<ApiError> {
  let payload: unknown;
  let rawBody = "";

  try {
    rawBody = (await response.text()).slice(0, MAX_ERROR_BODY_LENGTH);
    payload = rawBody ? JSON.parse(rawBody) : undefined;
  } catch {
    payload = rawBody;
  }

  const detail = errorMessageValue(payload);
  const statusText = response.statusText.trim();
  const suffix = detail || statusText;
  return new ApiError(
    `API request failed with HTTP ${response.status}${suffix ? `: ${suffix}` : ""}`,
    response.status,
    payload,
  );
}

async function assertOk(response: Response): Promise<Response> {
  if (!response.ok) throw await responseError(response);
  return response;
}

function normalizeThrownError(error: unknown, signal: AbortSignal): unknown {
  if (signal.aborted && signal.reason !== undefined) return signal.reason;
  if (
    error instanceof TypeError &&
    /fetch|network|failed to load|load failed/i.test(error.message)
  ) {
    return new ApiError(
      "The connection to KIRAG was interrupted. Check that the API and frontend services are running, then retry the query.",
      0,
      error,
    );
  }
  return error;
}

async function withApiResponse<T>(
  path: string,
  options: ApiRequestOptions,
  consume: (response: Response) => Promise<T>,
): Promise<T> {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, json, headers: inputHeaders, signal: externalSignal, ...init } = options;
  const scope = createRequestScope(externalSignal, timeoutMs);
  const headers = new Headers(inputHeaders);
  let body = options.body;

  if (json !== undefined) {
    if (body !== undefined && body !== null) {
      scope.close();
      throw new TypeError("Use either json or body, not both");
    }
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(json);
  }

  try {
    const response = await fetch(apiUrl(path), {
      ...init,
      body,
      headers,
      signal: scope.signal,
      cache: init.cache ?? "no-store",
      credentials: "same-origin",
    });
    await assertOk(response);
    return await consume(response);
  } catch (error) {
    throw normalizeThrownError(error, scope.signal);
  } finally {
    scope.close();
  }
}

export async function requestJson<T = unknown>(
  path: string,
  options: ApiRequestOptions = {},
): Promise<T> {
  const headers = new Headers(options.headers);
  if (!headers.has("Accept")) headers.set("Accept", "application/json");

  return withApiResponse(path, { ...options, headers }, async (response) => {
    if (response.status === 204) return undefined as T;
    try {
      return (await response.json()) as T;
    } catch (error) {
      throw new ApiError(
        `API returned invalid JSON with HTTP ${response.status}`,
        response.status,
        error,
      );
    }
  });
}

export async function requestText(
  path: string,
  options: ApiRequestOptions = {},
): Promise<string> {
  return withApiResponse(path, options, (response) => response.text());
}

export async function requestBlob(
  path: string,
  options: ApiRequestOptions = {},
): Promise<Blob> {
  return withApiResponse(path, options, (response) => response.blob());
}

/**
 * Incremental SSE parser implementing the event framing rules from the HTML
 * standard. Both UTF-8 code points and CRLF delimiters may be split across
 * arbitrary byte chunks.
 */
export class BufferedSseParser {
  private readonly decoder = new TextDecoder();
  private readonly onEvent: (event: ParsedSseEvent) => void;
  private textBuffer = "";
  private dataLines: string[] = [];
  private eventType: string | undefined;
  private lastEventId: string | undefined;
  private finished = false;

  constructor(onEvent: (event: ParsedSseEvent) => void) {
    this.onEvent = onEvent;
  }

  push(chunk: Uint8Array): void {
    if (this.finished) throw new TypeError("Cannot push bytes after the SSE parser is finished");
    this.textBuffer += this.decoder.decode(chunk, { stream: true });
    this.processLines(false);
  }

  finish(): void {
    if (this.finished) return;
    this.finished = true;
    this.textBuffer += this.decoder.decode();
    this.processLines(true);
    this.dispatchEvent();
  }

  private processLines(atEnd: boolean): void {
    let lineStart = 0;

    for (let index = 0; index < this.textBuffer.length; index += 1) {
      const char = this.textBuffer[index];
      if (char !== "\r" && char !== "\n") continue;
      if (char === "\r" && index + 1 === this.textBuffer.length && !atEnd) break;

      this.processLine(this.textBuffer.slice(lineStart, index));
      if (char === "\r" && this.textBuffer[index + 1] === "\n") index += 1;
      lineStart = index + 1;
    }

    this.textBuffer = this.textBuffer.slice(lineStart);
    if (atEnd && this.textBuffer) {
      this.processLine(this.textBuffer);
      this.textBuffer = "";
    }
  }

  private processLine(line: string): void {
    if (line === "") {
      this.dispatchEvent();
      return;
    }
    if (line.startsWith(":")) return;

    const colonIndex = line.indexOf(":");
    const field = colonIndex < 0 ? line : line.slice(0, colonIndex);
    let value = colonIndex < 0 ? "" : line.slice(colonIndex + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    if (field === "data") {
      this.dataLines.push(value);
    } else if (field === "event") {
      this.eventType = value;
    } else if (field === "id" && !value.includes("\0")) {
      this.lastEventId = value;
    }
  }

  private dispatchEvent(): void {
    if (this.dataLines.length === 0) {
      this.eventType = undefined;
      return;
    }

    const event: ParsedSseEvent = {
      data: this.dataLines.join("\n"),
      ...(this.eventType ? { event: this.eventType } : {}),
      ...(this.lastEventId !== undefined ? { id: this.lastEventId } : {}),
    };
    this.dataLines = [];
    this.eventType = undefined;
    this.onEvent(event);
  }
}

function parseJsonSseData<T>(rawData: string): { done: boolean; data?: T } {
  if (rawData.trim() === "[DONE]") return { done: true };

  try {
    return { done: false, data: JSON.parse(rawData) as T };
  } catch (error) {
    throw new ApiError("API returned invalid JSON in an SSE event", 502, error);
  }
}

function ssePayloadError(payload: unknown): ApiError | null {
  if (!payload || typeof payload !== "object") return null;
  const record = payload as Record<string, unknown>;
  if (!("error" in record)) return null;
  const message = errorMessageValue(payload) || "Streaming API request failed";
  return new ApiError(message, 502, payload);
}

async function consumeSse<T>(
  response: Response,
  handlers: SseHandlers<T>,
  signal?: AbortSignal,
): Promise<void> {
  if (!response.body) {
    throw new ApiError("API returned an empty streaming response", response.status);
  }

  const reader = response.body.getReader();
  const cancelReader = () => {
    void reader.cancel(signal?.reason).catch(() => undefined);
  };
  if (signal?.aborted) cancelReader();
  else signal?.addEventListener("abort", cancelReader, { once: true });
  let streamDone = false;
  const parser = new BufferedSseParser((rawEvent) => {
    if (streamDone) return;
    const event = parseJsonSseData<T>(rawEvent.data);
    if (event.done) {
      streamDone = true;
      return;
    }
    if (event.data !== undefined) {
      const apiError = ssePayloadError(event.data);
      if (apiError) throw apiError;
      handlers.onMessage(event.data);
    }
  });

  try {
    while (!streamDone) {
      const { done, value } = await reader.read();
      if (done) {
        parser.finish();
        if (!streamDone) {
          throw new ApiError(
            "The RAG stream ended before the server confirmed completion. Retry the query.",
            502,
          );
        }
        break;
      }
      parser.push(value);
    }
    // [DONE] is authoritative. Do not await reader.cancel(): some streaming
    // proxies leave its promise pending after the complete SSE event arrived.
  } finally {
    signal?.removeEventListener("abort", cancelReader);
    reader.releaseLock();
  }
}

export function requestJsonSse<T = unknown>(
  path: string,
  options: ApiRequestOptions,
  handlers: SseHandlers<T>,
): ApiRequestHandle {
  const controller = new AbortController();
  let cancelledByCaller = false;
  let settled = false;

  const finish = () => {
    if (settled) return;
    settled = true;
    handlers.onComplete();
  };

  const headers = new Headers(options.headers);
  headers.set("Accept", "text/event-stream");

  void withApiResponse(
    path,
    {
      ...options,
      headers,
      signal: controller.signal,
      timeoutMs: options.timeoutMs ?? API_TIMEOUTS.stream,
    },
    (response) => consumeSse(response, handlers, controller.signal),
  )
    .then(finish)
    .catch((error) => {
      if (!cancelledByCaller && !settled) {
        settled = true;
        handlers.onError(error);
      }
    });

  return {
    cancel: (reason = "API request cancelled") => {
      if (settled || controller.signal.aborted) return;
      cancelledByCaller = true;
      settled = true;
      controller.abort(new DOMException(reason, "AbortError"));
    },
  };
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}

export function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
