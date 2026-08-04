/** @jest-environment node */

import {
  ApiError,
  ApiTimeoutError,
  BufferedSseParser,
  apiPathSegment,
  apiUrl,
  requestJson,
  requestJsonSse,
} from "../api-client";

const originalFetch = global.fetch;

describe("same-origin API client", () => {
  afterEach(() => {
    global.fetch = originalFetch;
    jest.restoreAllMocks();
  });

  test("rejects absolute URLs and dot resource segments", () => {
    expect(() => apiUrl("https://api.example.test/api/health")).toThrow(TypeError);
    expect(() => apiPathSegment("..")).toThrow(TypeError);
    expect(apiUrl("/api/health")).toBe("/api/health");
  });

  test("creates same-origin JSON requests with centralized defaults", async () => {
    const fetchMock = jest.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    global.fetch = fetchMock;

    await expect(
      requestJson("/api/example", { method: "POST", json: { value: 1 } }),
    ).resolves.toEqual({ ok: true });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/example");
    expect(init.credentials).toBe("same-origin");
    expect(init.cache).toBe("no-store");
    expect(init.body).toBe(JSON.stringify({ value: 1 }));
    expect(new Headers(init.headers).get("Content-Type")).toBe("application/json");
  });

  test("normalizes JSON error responses", async () => {
    global.fetch = jest.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "Invalid or missing API key" }), {
        status: 401,
        statusText: "Unauthorized",
      }),
    );

    await expect(requestJson("/api/health")).rejects.toMatchObject<ApiError>({
      name: "ApiError",
      status: 401,
      message: "API request failed with HTTP 401: Invalid or missing API key",
    });
  });

  test("normalizes typed API error envelopes", async () => {
    global.fetch = jest.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          error: { code: "validation_error", message: "Request validation failed" },
        }),
        { status: 422 },
      ),
    );

    await expect(requestJson("/api/example")).rejects.toMatchObject<ApiError>({
      name: "ApiError",
      status: 422,
      message: "API request failed with HTTP 422: Request validation failed",
    });
  });

  test("aborts requests when the centralized timeout expires", async () => {
    global.fetch = jest.fn().mockImplementation((_url: string, init: RequestInit) => {
      return new Promise((_resolve, reject) => {
        init.signal?.addEventListener("abort", () => reject(init.signal?.reason), {
          once: true,
        });
      });
    });

    await expect(
      requestJson("/api/slow", { timeoutMs: 5 }),
    ).rejects.toBeInstanceOf(ApiTimeoutError);
  });

  test("normalizes browser fetch network failures into an actionable API error", async () => {
    global.fetch = jest.fn().mockRejectedValue(
      new TypeError("NetworkError when attempting to fetch resource."),
    );

    await expect(requestJson("/api/health")).rejects.toMatchObject<ApiError>({
      name: "ApiError",
      status: 0,
      message: expect.stringContaining("connection to KIRAG was interrupted"),
    });
  });

  test("normalizes Firefox input-stream failures during SSE consumption", async () => {
    const brokenStream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.error(new TypeError("Error in input stream"));
      },
    });
    global.fetch = jest.fn().mockResolvedValue(new Response(brokenStream, { status: 200 }));

    const error = await new Promise<unknown>((resolve) => {
      requestJsonSse("/api/stream", {}, {
        onMessage: jest.fn(),
        onError: resolve,
        onComplete: jest.fn(),
      });
    });

    expect(error).toMatchObject({
      name: "ApiError",
      status: 0,
      message: expect.stringContaining("connection to KIRAG was interrupted"),
    });
  });

  async function collectStream(byteChunks: Uint8Array[]): Promise<string[]> {
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        byteChunks.forEach((chunk) => controller.enqueue(chunk));
        controller.close();
      },
    });
    global.fetch = jest.fn().mockResolvedValue(
      new Response(stream, {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      }),
    );

    const messages: string[] = [];
    await new Promise<void>((resolve, reject) => {
      requestJsonSse<{ chunk: string }>(
        "/api/stream",
        { method: "POST", json: {} },
        {
          onMessage: (event) => messages.push(event.chunk),
          onError: reject,
          onComplete: resolve,
        },
      );
    });
    return messages;
  }

  test("reproduces the exact SSE response when split at every byte boundary", async () => {
    const encoder = new TextEncoder();
    const bytes = encoder.encode(
      ': keepalive\r\nevent: message\r\ndata: {"chunk":\r\ndata: "héllo 🌏"}\r\n\r\ndata: [DONE]\r\n\r\n',
    );

    for (let boundary = 1; boundary < bytes.length; boundary += 1) {
      const messages = await collectStream([
        bytes.slice(0, boundary),
        bytes.slice(boundary),
      ]);
      expect(messages.join("")).toBe("héllo 🌏");
    }

    const oneByteAtATime = Array.from(bytes, (_value, index) => bytes.slice(index, index + 1));
    await expect(collectStream(oneByteAtATime)).resolves.toEqual(["héllo 🌏"]);
  });

  test("completes on DONE even when transport cancellation would never settle", async () => {
    const done = new TextEncoder().encode('data: {"chunk":"complete"}\n\ndata: [DONE]\n\n');
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(done);
      },
      cancel() {
        return new Promise<void>(() => undefined);
      },
    });
    global.fetch = jest.fn().mockResolvedValue(new Response(stream, { status: 200 }));

    const completion = new Promise<string>((resolve, reject) => {
      requestJsonSse<{ chunk: string }>("/api/stream", {}, {
        onMessage: (event) => expect(event.chunk).toBe("complete"),
        onError: reject,
        onComplete: () => resolve("done"),
      });
    });

    await expect(completion).resolves.toBe("done");
  });

  test("treats the SSE timeout as inactivity rather than a total runtime limit", async () => {
    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      async start(controller) {
        for (let index = 0; index < 4; index += 1) {
          controller.enqueue(encoder.encode(": keep-alive\n\n"));
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      },
    });
    global.fetch = jest.fn().mockResolvedValue(new Response(stream, { status: 200 }));

    const completion = new Promise<string>((resolve, reject) => {
      requestJsonSse("/api/stream", { timeoutMs: 25 }, {
        onMessage: jest.fn(),
        onError: reject,
        onComplete: () => resolve("done"),
      });
    });

    await expect(completion).resolves.toBe("done");
  });

  test("the reusable parser accepts LF, CRLF, CR, comments, and multiple data lines", () => {
    const events: string[] = [];
    const parser = new BufferedSseParser((event) => events.push(event.data));
    const bytes = new TextEncoder().encode(
      ": ignored\rdata: first\r\ndata: second\n\nretry: 10\ndata: last",
    );
    Array.from(bytes, (_value, index) => bytes.slice(index, index + 1)).forEach((byte) =>
      parser.push(byte),
    );
    parser.finish();
    expect(events).toEqual(["first\nsecond", "last"]);
  });

  test("reports HTTP failures without completing", async () => {
    global.fetch = jest.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          error: { code: "service_unavailable", message: "Analysis service unavailable" },
        }),
        { status: 503 },
      ),
    );
    const onError = jest.fn();
    const onComplete = jest.fn();

    await new Promise<void>((resolve) => {
      requestJsonSse("/api/stream", {}, {
        onMessage: jest.fn(),
        onError: (error) => {
          onError(error);
          resolve();
        },
        onComplete,
      });
    });

    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({
        status: 503,
        message: "API request failed with HTTP 503: Analysis service unavailable",
      }),
    );
    expect(onComplete).not.toHaveBeenCalled();
  });

  test("reports reader and typed SSE errors without completing", async () => {
    const readerFailure = new Error("reader failed");
    const brokenStream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.error(readerFailure);
      },
    });
    global.fetch = jest.fn().mockResolvedValue(new Response(brokenStream, { status: 200 }));

    const firstError = await new Promise<unknown>((resolve) => {
      requestJsonSse("/api/stream", {}, {
        onMessage: jest.fn(),
        onError: resolve,
        onComplete: jest.fn(),
      });
    });
    expect(firstError).toBe(readerFailure);

    const encodedError = new TextEncoder().encode(
      'data: {"error":{"code":"rag_query_failed","message":"RAG query failed"}}\n\n',
    );
    const typedError = await new Promise<unknown>((resolve) => {
      void collectStream([encodedError]).catch(resolve);
    });
    expect(typedError).toMatchObject({
      name: "ApiError",
      status: 502,
      message: "RAG query failed",
    });
  });

  test("reports a prematurely closed SSE response instead of treating it as complete", async () => {
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('data: {"chunk":"partial"}\n\n'));
        controller.close();
      },
    });
    global.fetch = jest.fn().mockResolvedValue(new Response(stream, { status: 200 }));

    const error = await new Promise<unknown>((resolve) => {
      requestJsonSse("/api/stream", {}, {
        onMessage: jest.fn(),
        onError: resolve,
        onComplete: jest.fn(),
      });
    });

    expect(error).toMatchObject({
      name: "ApiError",
      status: 502,
      message: expect.stringContaining("ended before the server confirmed completion"),
    });
  });

  test("caller cancellation suppresses completion and error callbacks", async () => {
    const streamCancelled = jest.fn();
    const stream = new ReadableStream<Uint8Array>({
      pull() {
        return new Promise(() => undefined);
      },
      cancel: streamCancelled,
    });
    global.fetch = jest.fn().mockResolvedValue(new Response(stream, { status: 200 }));
    const onError = jest.fn();
    const onComplete = jest.fn();
    const handle = requestJsonSse("/api/stream", {}, {
      onMessage: jest.fn(),
      onError,
      onComplete,
    });

    await new Promise((resolve) => setTimeout(resolve, 0));
    handle.cancel("test cancellation");
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(streamCancelled).toHaveBeenCalledTimes(1);
    expect(onError).not.toHaveBeenCalled();
    expect(onComplete).not.toHaveBeenCalled();
  });
});
