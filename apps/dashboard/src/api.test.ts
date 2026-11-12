import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, getToken, setToken } from "./api";
import { mockApi, READY } from "./test/fixtures";

beforeEach(() => { sessionStorage.clear(); });

describe("api client", () => {
  it("sends the bearer token from session storage", async () => {
    setToken("secret-token");
    const { calls } = mockApi({ "GET /overview": { body: {} } });
    await api.overview();
    expect(calls[0].headers.Authorization).toBe("Bearer secret-token");
  });

  it("keeps the token only for the session and can clear it", () => {
    setToken("abc");
    expect(getToken()).toBe("abc");
    expect(localStorage.getItem("aegis.token")).toBeNull();
    setToken("");
    expect(getToken()).toBe("");
  });

  it("turns a structured API error into ApiError with code, message and request id", async () => {
    mockApi({ "GET /incidents/INC-9": { status: 404, body: { error: { code: "not_found", message: "incident 'INC-9' not found", request_id: "req-1" } } } });
    const error = await api.incident("INC-9").catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 404, code: "not_found", message: "incident 'INC-9' not found", requestId: "req-1" });
  });

  it("survives an error response that is not JSON", async () => {
    mockApi({ "GET /overview": { status: 502, text: "<html>Bad Gateway</html>" } });
    const error = await api.overview().catch((e) => e);
    expect(error).toMatchObject({ status: 502, code: "http_error" });
  });

  it("reports an unreachable API as status 0, not as an unhandled rejection", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    await expect(api.overview()).rejects.toMatchObject({ status: 0, code: "network", message: "The API is unreachable." });
    await expect(api.ready()).rejects.toMatchObject({ status: 0, code: "network" });
  });

  it("returns the readiness body even when the server answers 503", async () => {
    mockApi({ "GET /ready": { status: 503, body: { ...READY, status: "not_ready" } } });
    await expect(api.ready()).resolves.toMatchObject({ status: "not_ready" });
  });

  it("sends only version and comment when deciding, with an idempotency key", async () => {
    const { calls } = mockApi({ "POST /remediation/REM-1/approve": { body: {} } });
    await api.decide("REM-1", "approve", 4, "looks right");
    expect(calls[0].body).toEqual({ expected_version: 4, comment: "looks right" });
    expect(calls[0].headers["Idempotency-Key"]).toBe("REM-1-approve-4");
    expect(JSON.stringify(calls[0].body)).not.toMatch(/role|reviewer|approver/);
  });

  it("escapes the run name in evaluation requests", async () => {
    const { calls } = mockApi({ "GET /evaluation/runs/*": { body: {} } });
    await api.evaluationRun("../x y");
    expect(calls[0].path).toBe("/evaluation/runs/..%2Fx%20y");
  });
});
