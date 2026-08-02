const ADMIN_ROUTE_PATTERNS: ReadonlyArray<{
  methods: ReadonlySet<string>;
  pattern: RegExp;
}> = [
  { methods: new Set(["POST", "PUT", "PATCH", "DELETE"]), pattern: /^docker(?:\/|$)/ },
  { methods: new Set(["POST", "DELETE"]), pattern: /^diagnostics(?:\/|$)/ },
  { methods: new Set(["PUT"]), pattern: /^settings$/ },
  { methods: new Set(["POST"]), pattern: /^pipeline\/runs\/[^/]+\/cancel$/ },
  { methods: new Set(["POST"]), pattern: /^rag\/(?:cases\/delete|embedding\/purge-cache|index(?:-all)?(?:\/stream)?|infra\/(?:start|stop))$/ },
  { methods: new Set(["POST"]), pattern: /^rag\/export$/ },
  { methods: new Set(["POST"]), pattern: /^system\/shutdown$/ },
];

export function isAdministrativeRoute(method: string, path: string[]): boolean {
  const joinedPath = path.join("/");
  return ADMIN_ROUTE_PATTERNS.some(
    ({ methods, pattern }) => methods.has(method) && pattern.test(joinedPath),
  );
}

export function adminProxyEnabled(value = process.env.KIRAG_ENABLE_ADMIN_PROXY): boolean {
  return value?.trim().toLowerCase() === "true";
}
