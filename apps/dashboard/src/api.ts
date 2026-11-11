import type { AuditEvent, IncidentDetail, Incident, InvestigationDetail, InvestigationSummary, Overview, Page, Ready, Remediation } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "/api";
const TOKEN_KEY = "aegis.token";

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public requestId?: string) {
    super(message);
  }
}

// The token lives in sessionStorage: it is gone when the tab closes and is never written to disk.
export const getToken = () => { try { return sessionStorage.getItem(TOKEN_KEY) ?? ""; } catch { return ""; } };
export const setToken = (token: string) => { try { token ? sessionStorage.setItem(TOKEN_KEY, token) : sessionStorage.removeItem(TOKEN_KEY); } catch { /* storage unavailable */ } };

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(BASE + path, {
      ...init,
      headers: { ...(init.body ? { "Content-Type": "application/json" } : {}), Authorization: `Bearer ${getToken()}`, ...init.headers },
    });
  } catch {
    throw new ApiError(0, "network", "The API is unreachable.");
  }
  if (!response.ok) {
    let body: any = null;
    try { body = await response.json(); } catch { /* non-JSON error */ }
    const error = body?.error;
    throw new ApiError(response.status, error?.code ?? "http_error", error?.message ?? response.statusText, error?.request_id);
  }
  return response.json() as Promise<T>;
}

const post = <T,>(path: string, body: unknown = {}, headers: Record<string, string> = {}) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body), headers });

export const api = {
  ready: async (): Promise<Ready> => {
    // /ready answers 503 with a body when a required component is down; that body is the answer.
    const response = await fetch(BASE + "/ready").catch(() => null);
    if (!response) throw new ApiError(0, "network", "The API is unreachable.");
    return response.json();
  },
  overview: () => request<Overview>("/overview"),
  incidents: (limit: number, offset: number, status: string) =>
    request<Page<Incident>>(`/incidents?limit=${limit}&offset=${offset}${status ? `&status=${status}` : ""}`),
  incident: (id: string) => request<IncidentDetail>(`/incidents/${id}`),
  investigation: (id: string) => request<InvestigationDetail>(`/investigations/${id}`),
  remediation: (id: string) => request<Remediation>(`/remediation/${id}`),
  audit: (incidentId: string) => request<Page<AuditEvent>>(`/audit?incident_id=${incidentId}&limit=200`),
  investigate: (id: string, useLlm: boolean) => post<InvestigationSummary>(`/incidents/${id}/investigate`, { use_llm: useLlm }),
  cancel: (id: string) => post<InvestigationSummary>(`/investigations/${id}/cancel`),
  createRemediation: (incidentId: string) => post<Remediation>(`/incidents/${incidentId}/remediation`),
  validate: (id: string) => post<Remediation>(`/remediation/${id}/validate`),
  // The body carries only what the reviewer saw (version) and said (comment). Who they are and
  // whether they may decide is established by the server from the token.
  decide: (id: string, decision: "approve" | "reject", expectedVersion: number, comment: string) =>
    post<Remediation>(`/remediation/${id}/${decision}`, { expected_version: expectedVersion, comment },
      { "Idempotency-Key": `${id}-${decision}-${expectedVersion}` }),
  rollback: (id: string, reason: string) => post<Remediation>(`/remediation/${id}/rollback`, { reason }),
};
