import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { ApiError } from "./api";

/** Load data, optionally polling. Distinguishes loading, error and data; never invents a value. */
export function useAsync<T>(load: () => Promise<T>, deps: unknown[], pollMs = 0) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const loader = useRef(load);
  loader.current = load;
  const reload = useCallback(async () => {
    try {
      setData(await loader.current());
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError(0, "unknown", String(e)));
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    setLoading(true);
    void reload();
    if (!pollMs) return;
    const timer = setInterval(() => void reload(), pollMs);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, pollMs, reload]);
  return { data, error, loading, reload };
}

export function StateBox({ loading, error, empty, emptyText, children }: {
  loading: boolean; error: ApiError | null; empty?: boolean; emptyText?: string; children: ReactNode;
}) {
  if (error) {
    const hint = error.status === 401 ? "Enter a valid API token (top right)."
      : error.status === 403 ? "Your role does not allow this."
      : error.status === 0 ? "Is the API running?" : "";
    return (
      <div className="state state-error" role="alert">
        <strong>{error.status === 0 ? "API unavailable" : `Error ${error.status}`}</strong>
        <span>{error.message} {hint}</span>
        {error.requestId && <code>request {error.requestId}</code>}
      </div>
    );
  }
  if (loading) return <div className="state" aria-busy="true">Loading…</div>;
  if (empty) return <div className="state">{emptyText ?? "Nothing to show."}</div>;
  return <>{children}</>;
}

// Status colours never carry meaning alone: every badge has an icon and a text label.
const SEVERITY: Record<string, { icon: string; tone: string }> = {
  critical: { icon: "▲", tone: "critical" }, high: { icon: "▲", tone: "serious" },
  medium: { icon: "◆", tone: "warning" }, low: { icon: "●", tone: "neutral" }, info: { icon: "●", tone: "neutral" },
};
export const Severity = ({ value }: { value: string }) => {
  const s = SEVERITY[value] ?? SEVERITY.info;
  return <span className={`badge tone-${s.tone}`}><i aria-hidden>{s.icon}</i>{value}</span>;
};

const STATUS_TONE: Record<string, string> = {
  open: "serious", investigating: "info", investigated: "warning", awaiting_approval: "warning",
  mitigated: "good", executed_unverified: "serious", running: "info", completed: "good", cancelled: "neutral",
  failed: "critical", ok: "good", ready: "good", unavailable: "serious", disabled: "neutral", not_ready: "critical",
  PROPOSED: "info", VALIDATING: "info", VALIDATED: "info", VALIDATION_FAILED: "serious", PENDING_APPROVAL: "warning",
  APPROVED: "info", REJECTED: "neutral", EXECUTED: "warning", COMPLETED: "good", ROLLED_BACK: "neutral",
  rejected: "serious", error: "critical", timeout: "serious", denied: "serious", available: "good", degraded: "warning",
};
const TONE_ICON: Record<string, string> = { good: "✓", warning: "◆", serious: "▲", critical: "▲", info: "◌", neutral: "●" };
export const Status = ({ value }: { value: string | null | undefined }) => {
  const v = value ?? "unknown";
  const tone = STATUS_TONE[v] ?? "neutral";
  return <span className={`badge tone-${tone}`}><i aria-hidden>{TONE_ICON[tone]}</i>{v.replace(/_/g, " ").toLowerCase()}</span>;
};

export const fmtTime = (iso: string | null | undefined) => (iso ? new Date(iso).toLocaleString() : "—");
export const fmtMs = (ms: number | null | undefined) =>
  ms == null ? "—" : ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms.toFixed(ms < 10 ? 1 : 0)} ms`;
export const fmtNum = (v: unknown, digits = 3) =>
  typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(digits)) : v == null ? "—" : String(v);

export const Card = ({ title, aside, children }: { title?: string; aside?: ReactNode; children: ReactNode }) => (
  <section className="card">
    {(title || aside) && <header><h3>{title}</h3>{aside}</header>}
    {children}
  </section>
);

export const Tile = ({ label, value, note }: { label: string; value: ReactNode; note?: string }) => (
  <div className="tile"><span className="tile-label">{label}</span><span className="tile-value">{value}</span>
    {note && <span className="tile-note">{note}</span>}</div>
);

/** Reference vs current for one metric: two thin bars on a shared 0..1 axis, values as text. */
export function MetricBars({ rows }: { rows: { name: string; reference: number | null; current: number | null }[] }) {
  return (
    <div className="metric-bars">
      <div className="legend"><span><i className="swatch s1" />reference</span><span><i className="swatch s2" />current</span></div>
      {rows.map((r) => (
        <div className="metric-row" key={r.name}>
          <span className="metric-name">{r.name}</span>
          <div className="metric-track" title={`${r.name}: reference ${fmtNum(r.reference)}, current ${fmtNum(r.current)}`}>
            <div className="bar s1" style={{ width: `${(r.reference ?? 0) * 100}%` }} />
            {r.current != null && <div className="bar s2" style={{ width: `${r.current * 100}%` }} />}
          </div>
          <span className="metric-values">{fmtNum(r.reference)} → {r.current == null ? "not measurable" : fmtNum(r.current)}</span>
        </div>
      ))}
    </div>
  );
}
