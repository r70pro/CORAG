import { adminProxyEnabled, isAdministrativeRoute } from "../admin-proxy-policy";

describe("administrative proxy policy", () => {
  it("is fail-closed unless explicitly enabled", () => {
    expect(adminProxyEnabled(undefined)).toBe(false);
    expect(adminProxyEnabled("false")).toBe(false);
    expect(adminProxyEnabled(" TRUE ")).toBe(true);
  });

  it.each([
    ["PUT", ["settings"]],
    ["POST", ["docker", "analysis", "switch"]],
    ["POST", ["rag", "index-all", "stream"]],
    ["DELETE", ["diagnostics", "models"]],
    ["POST", ["system", "shutdown"]],
  ])("recognizes %s /api/%s as administrative", (method, path) => {
    expect(isAdministrativeRoute(method, path)).toBe(true);
  });

  it.each([
    ["GET", ["settings"]],
    ["GET", ["docker", "status"]],
    ["POST", ["rag", "query"]],
    ["GET", ["documents", "runs"]],
  ])("does not elevate %s /api/%s", (method, path) => {
    expect(isAdministrativeRoute(method, path)).toBe(false);
  });
});
