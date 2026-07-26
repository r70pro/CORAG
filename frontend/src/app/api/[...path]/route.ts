import "server-only";

import type { NextRequest } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const DEFAULT_API_URL = "http://127.0.0.1:8001";
const REQUEST_HEADER_ALLOWLIST = [
  "accept",
  "accept-language",
  "content-length",
  "content-type",
  "if-match",
  "if-modified-since",
  "if-none-match",
  "if-range",
  "if-unmodified-since",
  "range",
] as const;
const RESPONSE_HEADER_ALLOWLIST = [
  "accept-ranges",
  "cache-control",
  "content-disposition",
  "content-language",
  "content-length",
  "content-range",
  "content-type",
  "etag",
  "expires",
  "last-modified",
  "location",
  "retry-after",
  "vary",
  "x-accel-buffering",
] as const;

type RouteContext = {
  params: Promise<{ path: string[] }>;
};

type UploadLimits = {
  maxFiles: number;
  maxFileBytes: number;
  maxTotalBytes: number;
  maxRequestBytes: number;
};

class UploadPolicyError extends Error {
  readonly status: 413 | 415;

  constructor(status: 413 | 415, message: string) {
    super(message);
    this.name = "UploadPolicyError";
    this.status = status;
  }
}

function positiveEnv(name: string, fallback: number, legacyName?: string): number {
  const raw = process.env[name] ?? (legacyName ? process.env[legacyName] : undefined);
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return value;
}

function uploadLimits(path: string[]): UploadLimits | null {
  let maxFiles: number;
  let maxFileBytes: number;
  let maxTotalBytes: number;

  if (path.join("/") === "pipeline/upload") {
    maxFiles = positiveEnv("KIRAG_MAX_PDF_FILES", 20);
    maxFileBytes = positiveEnv(
      "KIRAG_MAX_PDF_FILE_BYTES",
      100 * 1024 * 1024,
      "KIRAG_MAX_UPLOAD_BYTES",
    );
    maxTotalBytes = positiveEnv(
      "KIRAG_MAX_PDF_UPLOAD_BYTES",
      500 * 1024 * 1024,
      "KIRAG_MAX_UPLOAD_TOTAL_BYTES",
    );
  } else if (path.join("/") === "rag/upload-markdown") {
    maxFiles = positiveEnv("KIRAG_MAX_MARKDOWN_FILES", 20);
    maxFileBytes = positiveEnv(
      "KIRAG_MAX_MARKDOWN_FILE_BYTES",
      10 * 1024 * 1024,
      "KIRAG_MAX_UPLOAD_BYTES",
    );
    maxTotalBytes = positiveEnv(
      "KIRAG_MAX_MARKDOWN_UPLOAD_BYTES",
      50 * 1024 * 1024,
      "KIRAG_MAX_UPLOAD_TOTAL_BYTES",
    );
  } else {
    return null;
  }

  return {
    maxFiles,
    maxFileBytes,
    maxTotalBytes,
    maxRequestBytes: maxTotalBytes + Math.max(1024 * 1024, maxFiles * 16 * 1024),
  };
}

function multipartBoundary(contentType: string): string | null {
  if (!contentType.toLowerCase().startsWith("multipart/form-data")) return null;
  const match = contentType.match(/\bboundary=(?:"([^"]+)"|([^;\s]+))/i);
  const boundary = match?.[1] || match?.[2] || "";
  return boundary && boundary.length <= 200 ? boundary : null;
}

function bytesIndexOf(haystack: Uint8Array, needle: Uint8Array): number {
  outer: for (let index = 0; index <= haystack.length - needle.length; index += 1) {
    for (let offset = 0; offset < needle.length; offset += 1) {
      if (haystack[index + offset] !== needle[offset]) continue outer;
    }
    return index;
  }
  return -1;
}

function bytesStartWith(value: Uint8Array, prefix: Uint8Array): boolean {
  if (value.length < prefix.length) return false;
  return prefix.every((byte, index) => value[index] === byte);
}

class MultipartUploadLimiter {
  private readonly firstBoundary: Uint8Array;
  private readonly bodyBoundary: Uint8Array;
  private readonly headerEnd = new TextEncoder().encode("\r\n\r\n");
  private readonly crlf = new TextEncoder().encode("\r\n");
  private readonly finalSuffix = new TextEncoder().encode("--");
  private readonly decoder = new TextDecoder("latin1");
  private buffer = new Uint8Array();
  private state: "initial" | "headers" | "body" | "suffix" | "end" = "initial";
  private requestBytes = 0;
  private totalFileBytes = 0;
  private currentFileBytes = 0;
  private fileCount = 0;
  private currentPartIsFile = false;

  constructor(boundary: string, private readonly limits: UploadLimits) {
    const encoded = new TextEncoder().encode(`--${boundary}`);
    this.firstBoundary = encoded;
    this.bodyBoundary = new TextEncoder().encode(`\r\n--${boundary}`);
  }

  private failSize(message: string): never {
    throw new UploadPolicyError(413, message);
  }

  private append(chunk: Uint8Array): void {
    const merged = new Uint8Array(this.buffer.length + chunk.length);
    merged.set(this.buffer);
    merged.set(chunk, this.buffer.length);
    this.buffer = merged;
  }

  private discard(length: number): void {
    this.buffer = this.buffer.slice(length);
  }

  private countBodyBytes(length: number): void {
    if (!this.currentPartIsFile || length <= 0) return;
    this.currentFileBytes += length;
    this.totalFileBytes += length;
    if (this.currentFileBytes > this.limits.maxFileBytes) {
      this.failSize("An uploaded file exceeds the per-file limit");
    }
    if (this.totalFileBytes > this.limits.maxTotalBytes) {
      this.failSize("The aggregate upload exceeds the size limit");
    }
  }

  feed(chunk: Uint8Array): void {
    this.requestBytes += chunk.byteLength;
    if (this.requestBytes > this.limits.maxRequestBytes) {
      this.failSize("The upload request exceeds the aggregate size limit");
    }
    this.append(chunk);

    while (this.state !== "end") {
      if (this.state === "initial") {
        if (this.buffer.length < this.firstBoundary.length + 2) return;
        if (!bytesStartWith(this.buffer, this.firstBoundary)) {
          throw new UploadPolicyError(415, "Malformed multipart upload");
        }
        this.discard(this.firstBoundary.length);
        if (!bytesStartWith(this.buffer, this.crlf)) {
          throw new UploadPolicyError(415, "Malformed multipart upload");
        }
        this.discard(this.crlf.length);
        this.state = "headers";
        continue;
      }

      if (this.state === "headers") {
        const headerIndex = bytesIndexOf(this.buffer, this.headerEnd);
        if (headerIndex < 0) {
          if (this.buffer.length > 16 * 1024) {
            throw new UploadPolicyError(415, "Multipart part headers are too large");
          }
          return;
        }
        const headers = this.decoder.decode(this.buffer.slice(0, headerIndex));
        this.discard(headerIndex + this.headerEnd.length);
        this.currentPartIsFile =
          /(?:^|\r\n)content-disposition:[^\r\n]*\bfilename\*?\s*=/i.test(headers);
        this.currentFileBytes = 0;
        if (this.currentPartIsFile) {
          this.fileCount += 1;
          if (this.fileCount > this.limits.maxFiles) {
            this.failSize("The upload contains too many files");
          }
        }
        this.state = "body";
        continue;
      }

      if (this.state === "body") {
        const boundaryIndex = bytesIndexOf(this.buffer, this.bodyBoundary);
        if (boundaryIndex < 0) {
          const safeLength = this.buffer.length - (this.bodyBoundary.length - 1);
          if (safeLength <= 0) return;
          this.countBodyBytes(safeLength);
          this.discard(safeLength);
          return;
        }
        this.countBodyBytes(boundaryIndex);
        this.discard(boundaryIndex + this.bodyBoundary.length);
        this.state = "suffix";
        continue;
      }

      if (this.state === "suffix") {
        if (this.buffer.length < 2) return;
        if (bytesStartWith(this.buffer, this.finalSuffix)) {
          this.discard(this.finalSuffix.length);
          this.state = "end";
          continue;
        }
        if (!bytesStartWith(this.buffer, this.crlf)) {
          throw new UploadPolicyError(415, "Malformed multipart boundary");
        }
        this.discard(this.crlf.length);
        this.state = "headers";
      }
    }
  }

  finish(): void {
    if (this.state !== "end") {
      throw new UploadPolicyError(415, "Incomplete multipart upload");
    }
  }
}

function jsonError(status: number, message: string): Response {
  return Response.json(
    {
      error: {
        code: status === 413 ? "payload_too_large" : "proxy_request_failed",
        message,
      },
    },
    {
      status,
      headers: {
        "Cache-Control": "no-store",
      },
    },
  );
}

function configuredApiBase(): URL {
  const rawUrl = process.env.KIRAG_API_URL?.trim() || DEFAULT_API_URL;
  const url = new URL(rawUrl);

  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password) {
    throw new Error("Invalid KIRAG_API_URL");
  }

  url.hash = "";
  url.search = "";
  return url;
}

function externalHost(request: NextRequest): string | null {
  const forwardedHost = request.headers.get("x-forwarded-host")?.split(",", 1)[0]?.trim();
  return forwardedHost || request.headers.get("host");
}

function isSameOriginRequest(request: NextRequest): boolean {
  if (request.headers.get("sec-fetch-site") === "cross-site") return false;

  const origin = request.headers.get("origin");
  if (!origin) return true;

  try {
    const host = externalHost(request);
    return Boolean(host && new URL(origin).host === host);
  } catch {
    return false;
  }
}

function upstreamUrl(request: NextRequest, path: string[], apiBase: URL): URL {
  const basePath = apiBase.pathname.replace(/\/+$/, "");
  const encodedPath = path.map((segment) => encodeURIComponent(segment)).join("/");
  apiBase.pathname = `${basePath}/api/${encodedPath}`;
  apiBase.search = request.nextUrl.search;
  return apiBase;
}

function upstreamHeaders(request: NextRequest, apiKey: string): Headers {
  const headers = new Headers();
  for (const name of REQUEST_HEADER_ALLOWLIST) {
    const value = request.headers.get(name);
    if (value !== null) headers.set(name, value);
  }

  headers.set("Accept-Encoding", "identity");
  headers.set("X-API-Key", apiKey);

  const adminApiKey = process.env.KIRAG_ADMIN_API_KEY?.trim();
  if (adminApiKey) headers.set("X-Admin-API-Key", adminApiKey);
  return headers;
}

function clientResponse(upstream: Response, apiBase: URL): Response {
  const headers = new Headers();
  for (const name of RESPONSE_HEADER_ALLOWLIST) {
    const value = upstream.headers.get(name);
    if (value !== null) headers.set(name, value);
  }

  const contentType = headers.get("content-type")?.toLowerCase() || "";
  if (contentType.includes("text/event-stream")) {
    headers.set("Cache-Control", "no-cache, no-transform");
    headers.set("X-Accel-Buffering", "no");
  }

  const location = headers.get("location");
  if (location) {
    try {
      const redirectUrl = new URL(location, apiBase);
      if (redirectUrl.origin === apiBase.origin && redirectUrl.pathname.startsWith("/api/")) {
        headers.set("location", `${redirectUrl.pathname}${redirectUrl.search}${redirectUrl.hash}`);
      } else {
        headers.delete("location");
      }
    } catch {
      headers.delete("location");
    }
  }

  const hasBody = ![204, 304].includes(upstream.status);
  return new Response(hasBody ? upstream.body : null, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers,
  });
}

async function proxyRequest(request: NextRequest, context: RouteContext): Promise<Response> {
  if (!isSameOriginRequest(request)) {
    return jsonError(403, "Cross-origin API requests are not allowed");
  }

  const apiKey = process.env.KIRAG_API_KEY?.trim();
  if (!apiKey) {
    return jsonError(503, "API proxy authentication is not configured");
  }

  let apiBase: URL;
  try {
    apiBase = configuredApiBase();
  } catch {
    return jsonError(503, "API proxy destination is not configured correctly");
  }

  const { path } = await context.params;
  const method = request.method.toUpperCase();
  const hasRequestBody = method !== "GET" && method !== "HEAD";
  let body: BodyInit | null | undefined = hasRequestBody ? request.body : undefined;
  const uploadState: { failure?: UploadPolicyError } = {};

  if (method === "POST") {
    let limits: UploadLimits | null;
    try {
      limits = uploadLimits(path);
    } catch {
      return jsonError(503, "Upload limits are not configured correctly");
    }
    if (limits) {
      const contentType = request.headers.get("content-type") || "";
      const boundary = multipartBoundary(contentType);
      if (!boundary) return jsonError(415, "A multipart/form-data upload is required");

      const declaredLength = Number(request.headers.get("content-length"));
      if (
        Number.isFinite(declaredLength) &&
        declaredLength > limits.maxRequestBytes
      ) {
        return jsonError(413, "The upload request exceeds the aggregate size limit");
      }
      if (!body) return jsonError(415, "The multipart upload has no body");

      const limiter = new MultipartUploadLimiter(boundary, limits);
      body = body.pipeThrough(
        new TransformStream<Uint8Array<ArrayBuffer>, Uint8Array<ArrayBuffer>>({
          transform(chunk, controller) {
            try {
              limiter.feed(chunk);
              controller.enqueue(chunk);
            } catch (error) {
              if (error instanceof UploadPolicyError) uploadState.failure = error;
              throw error;
            }
          },
          flush() {
            try {
              limiter.finish();
            } catch (error) {
              if (error instanceof UploadPolicyError) uploadState.failure = error;
              throw error;
            }
          },
        }),
      );
    }
  }

  const init: RequestInit & { duplex?: "half" } = {
    method,
    headers: upstreamHeaders(request, apiKey),
    body,
    cache: "no-store",
    redirect: "manual",
    signal: request.signal,
  };
  if (hasRequestBody) init.duplex = "half";

  try {
    const upstream = await fetch(upstreamUrl(request, path, new URL(apiBase)), init);
    return clientResponse(upstream, apiBase);
  } catch {
    const uploadFailure = uploadState.failure;
    if (uploadFailure) {
      return jsonError(uploadFailure.status, uploadFailure.message);
    }
    if (request.signal.aborted) {
      return jsonError(499, "Client closed the API request");
    }
    return jsonError(502, "The API service is unavailable");
  }
}

export const GET = proxyRequest;
export const POST = proxyRequest;
export const PUT = proxyRequest;
export const PATCH = proxyRequest;
export const DELETE = proxyRequest;
export const HEAD = proxyRequest;

export function OPTIONS(): Response {
  return new Response(null, {
    status: 204,
    headers: {
      Allow: "GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS",
      "Cache-Control": "no-store",
    },
  });
}
