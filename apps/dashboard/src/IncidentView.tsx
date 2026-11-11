import { useState } from "react";
import { ApiError, api } from "./api";
import type { Evidence, IncidentDetail, InvestigationDetail, Remediation } from "./types";
import { Card, MetricBars, Severity, StateBox, Status, fmtMs, fmtNum, fmtTime, useAsync } from "./ui";

const TABS = ["Evidence", "Investigation", "Remediation", "Audit trail"] as const;
type Tab = (typeof TABS)[number];

export function IncidentView({ id }: { id: string }) {
  const [tab, setTab] = useState<Tab>("Evidence");
  const incident = useAsync(() => api.incident(id), [id], 5000);
  const d = incident.data;
  return (
    <StateBox loading={incident.loading && !d} error={incident.error}>
      {d && (
        <>
          <div className="page-head">
            <div><a href="#/incidents" className="back">← Incidents</a><h2><code>{d.id}</code></h2><p className="muted">{d.summary}</p></div>
            <dl className="kv compact">
              <dt>Severity</dt><dd><Severity value={d.severity} /></dd>
              <dt>Status</dt><dd><Status value={d.status} /></dd>
              <dt>Detected</dt><dd>{fmtTime(d.detected_at)}</dd>
              <dt>Model</dt><dd><code>{d.model_version}</code></dd>
              <dt>Reference</dt><dd><code>{d.reference_version}</code></dd>
            </dl>
          </div>
          <div className="tabs" role="tablist">
            {TABS.map((t) => <button key={t} role="tab" aria-selected={t === tab} className={t === tab ? "active" : ""} onClick={() => setTab(t)}>{t}</button>)}
          </div>
          {tab === "Evidence" && <EvidenceTab d={d} />}
          {tab === "Investigation" && <InvestigationTab d={d} onChange={incident.reload} />}
          {tab === "Remediation" && <RemediationTab d={d} onChange={incident.reload} />}
          {tab === "Audit trail" && <AuditTab id={d.id} />}
        </>
      )}
    </StateBox>
  );
}

const LEVEL: Record<string, string> = { observed_fact: "observed fact", statistical_anomaly: "statistical anomaly" };

function EvidenceTab({ d }: { d: IncidentDetail }) {
  const features = d.findings.filter((f) => f.feature);
  const perf = d.performance;
  return (
    <>
      <Card title="Monitoring findings" aside={<span className="muted">what was measured — not what caused it</span>}>
        {d.findings.length === 0 ? <p className="muted">No triggered findings.</p> : (
          <table><thead><tr><th>Check</th><th>Feature</th><th>Observed</th><th>Reference</th><th>Threshold</th><th>Severity</th><th>Kind of claim</th></tr></thead>
            <tbody>{d.findings.map((f) => (
              <tr key={f.finding_id} title={f.interpretation}>
                <td>{f.monitor} · {f.check}</td><td>{f.feature ?? "—"}</td><td>{fmtNum(f.observed_value)}</td>
                <td>{fmtNum(f.reference_value)}</td><td>{fmtNum(f.threshold)}</td><td><Severity value={f.severity} /></td>
                <td>{LEVEL[f.level] ?? f.level}</td>
              </tr>))}</tbody></table>
        )}
        {d.skipped_checks.length > 0 && (
          <details><summary>{d.skipped_checks.length} check(s) could not run</summary>
            <ul>{d.skipped_checks.map((s, i) => <li key={i}><b>{s.monitor} · {s.check}</b>{s.feature ? ` (${s.feature})` : ""}: {s.reason}</li>)}</ul>
          </details>
        )}
      </Card>
      <div className="grid-2">
        <Card title="Feature-level findings">
          {features.length === 0 ? <p className="muted">No feature-specific findings.</p>
            : <ul className="plain">{features.map((f) => <li key={f.finding_id}><b>{f.feature}</b> — {f.interpretation}</li>)}</ul>}
        </Card>
        <Card title="Performance change">
          {!perf.reference ? <p className="muted">Unavailable: the batch could not be scored.</p> : (
            <>
              <MetricBars rows={["accuracy", "precision", "recall", "f1", "roc_auc"].map((m) => ({
                name: m, reference: perf.reference![m] ?? null, current: perf.current ? perf.current[m] ?? null : null }))} />
              {!perf.current && <p className="muted">Current metrics are not measurable ({perf.n_labelled} labelled rows).</p>}
            </>
          )}
        </Card>
      </div>
      <Card title={`Evidence records (${d.evidence.length})`}>
        <EvidenceTable evidence={d.evidence} />
      </Card>
    </>
  );
}

function EvidenceTable({ evidence }: { evidence: Evidence[] }) {
  return (
    <table><thead><tr><th>ID</th><th>Source</th><th>Polarity</th><th>Statement</th></tr></thead>
      <tbody>{evidence.map((e) => (
        <tr key={e.evidence_id}><td><code>{e.evidence_id}</code></td><td>{e.source}</td>
          <td>{e.data.polarity === "anomaly" ? <Severity value={e.severity} /> : <span className="muted">normal</span>}</td>
          <td>{e.statement}</td></tr>))}</tbody></table>
  );
}

function useAction(onDone: () => void) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true); setError(null);
    try { await fn(); onDone(); } catch (e) { setError(e instanceof ApiError ? `${e.message} (${e.status})` : String(e)); } finally { setBusy(false); }
  };
  return { busy, error, run };
}

function InvestigationTab({ d, onChange }: { d: IncidentDetail; onChange: () => void }) {
  const latest = d.investigations[0];
  const action = useAction(onChange);
  const [useLlm, setUseLlm] = useState(true);
  const detail = useAsync<InvestigationDetail | null>(() => (latest ? api.investigation(latest.id) : Promise.resolve(null)),
    [latest?.id, latest?.status], latest?.status === "running" ? 3000 : 0);
  const inv = detail.data;
  const statements = new Map(d.evidence.map((e) => [e.evidence_id, e.statement]));
  return (
    <>
      <Card title="Agent investigation" aside={
        <span className="actions">
          <label><input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} /> use local LLM</label>
          <button disabled={action.busy || latest?.status === "running"} onClick={() => action.run(() => api.investigate(d.id, useLlm))}>
            {latest ? "Investigate again" : "Start investigation"}</button>
          {latest?.status === "running" && <button className="ghost" onClick={() => action.run(() => api.cancel(latest.id))}>Cancel</button>}
        </span>}>
        {action.error && <p className="error" role="alert">{action.error}</p>}
        {!latest ? <p className="muted">This incident has not been investigated yet.</p> : (
          <StateBox loading={detail.loading && !inv} error={detail.error}>
            {inv && (
              <>
                <dl className="kv">
                  <dt>State</dt><dd><Status value={inv.status} />{inv.status === "running" && " — CPU inference can take several minutes"}</dd>
                  <dt>LLM status</dt><dd><Status value={inv.llm_status} /> {inv.llm_model ? <code>{inv.llm_model}</code> : <span className="muted">no model used</span>}</dd>
                  <dt>Duration</dt><dd>{fmtMs(inv.duration_ms)}</dd>
                  <dt>Requested by</dt><dd>{inv.requested_by} · {fmtTime(inv.started_at)}</dd>
                </dl>
                {inv.errors.length > 0 && <ul className="notes">{inv.errors.map((e, i) => <li key={i}>{e}</li>)}</ul>}
                {inv.explanation && (
                  <div className="explanation">
                    <h4>Final explanation <span className="muted">(source: {inv.explanation.source})</span></h4>
                    <p className="pre">{inv.explanation.summary}</p>
                    {inv.explanation.operator_checks.length > 0 && <ul>{inv.explanation.operator_checks.map((c, i) => <li key={i}>{c}</li>)}</ul>}
                  </div>
                )}
              </>
            )}
          </StateBox>
        )}
      </Card>
      {inv && inv.status !== "running" && (
        <>
          <Card title="Root-cause hypotheses" aside={<span className="muted">ranked by evidence fit — hypotheses, not established causes</span>}>
            {inv.diagnosis && <p><b>{inv.diagnosis.conclusion}</b></p>}
            {inv.hypotheses.length === 0 ? <p className="muted">No hypothesis is supported by the evidence.</p> : inv.hypotheses.map((h) => (
              <div className="hypothesis" key={h.hypothesis_id}>
                <div className="hyp-head">
                  <span className="rank">#{h.rank}</span><b>{h.root_cause.replace(/_/g, " ")}</b>
                  <div className="fit" title={`evidence fit ${h.confidence.toFixed(2)} (not a probability of causation)`}><div style={{ width: `${h.confidence * 100}%` }} /></div>
                  <span>fit {h.confidence.toFixed(2)}</span><span className="muted">ranked by {h.ranked_by}</span>
                </div>
                <p>{h.payload.reasoning}</p>
                <EvidenceRefs label="Supporting evidence" ids={h.supporting_evidence_ids} statements={statements} />
                <EvidenceRefs label="Contradicting evidence" ids={h.contradicting_evidence_ids} statements={statements} />
                {h.payload.missing_evidence?.length > 0 && <p className="muted">Missing: {h.payload.missing_evidence.join("; ")}</p>}
              </div>
            ))}
          </Card>
          <div className="grid-2">
            <Card title={`Tool calls (${inv.tool_traces.length})`}>
              <table><thead><tr><th>#</th><th>Tool</th><th>By</th><th>Status</th><th>Duration</th></tr></thead>
                <tbody>{inv.tool_traces.map((t) => (
                  <tr key={t.id} title={t.error ?? t.output?.summary ?? ""}>
                    <td>{t.seq + 1}</td><td><code>{t.tool}</code><div className="muted small">{t.error ?? t.output?.summary}</div></td>
                    <td>{t.requested_by}</td><td><Status value={t.status} /></td><td>{fmtMs(t.duration_ms)}</td></tr>))}</tbody></table>
            </Card>
            <Card title="LLM requests and workflow">
              {inv.llm_calls.length === 0 ? <p className="muted">No LLM requests were made; the deterministic path produced this result.</p> : (
                <table><thead><tr><th>Purpose</th><th>Outcome</th><th>Duration</th><th>Tokens in/out</th></tr></thead>
                  <tbody>{inv.llm_calls.map((c, i) => (
                    <tr key={i}><td>{c.purpose}</td><td><Status value={c.ok ? "ok" : c.error ?? "error"} /></td><td>{fmtMs(c.duration_ms)}</td>
                      <td>{c.prompt_tokens ?? "—"} / {c.completion_tokens ?? "—"}</td></tr>))}</tbody></table>
              )}
              <h4>Workflow steps</h4>
              <ol className="steps">{inv.events.map((e, i) => <li key={i} className={e.status}><span>{e.node.replace(/_/g, " ")}</span><span className="muted">{e.status === "ok" ? fmtMs(e.duration_ms) : e.status}</span></li>)}</ol>
            </Card>
          </div>
          <Card title="Runbook references" aside={<span className="muted">reference text only; runbooks cannot authorise actions</span>}>
            {inv.runbook_refs.length === 0 ? <p className="muted">No runbook passages were retrieved.</p> : inv.runbook_refs.map((r) => (
              <details key={r.passage_id}><summary><code>{r.passage_id}</code> {r.doc_title} — {r.section} <span className="muted">(v{r.doc_version}, score {r.score.toFixed(2)})</span>
                {r.suspicious && <span className="badge tone-serious"><i aria-hidden>▲</i>flagged: instruction-like text</span>}</summary><p>{r.text}</p></details>
            ))}
          </Card>
        </>
      )}
    </>
  );
}

function EvidenceRefs({ label, ids, statements }: { label: string; ids: string[]; statements: Map<string, string> }) {
  if (ids.length === 0) return <p className="muted small">{label}: none</p>;
  return (
    <details><summary>{label} ({ids.length})</summary>
      <ul className="plain small">{ids.map((i) => <li key={i}><code>{i}</code> {statements.get(i) ?? ""}</li>)}</ul></details>
  );
}

function RemediationTab({ d, onChange }: { d: IncidentDetail; onChange: () => void }) {
  const action = useAction(onChange);
  const recommendation = d.investigations.length > 0;
  if (d.remediations.length === 0) {
    return (
      <Card title="Remediation">
        <p className="muted">{recommendation
          ? "No state-changing remediation is on record for this incident. If the investigation recommended no action or further investigation, that is the outcome."
          : "Investigate the incident first; remediation proposals are derived from the diagnosis."}</p>
        {recommendation && <button disabled={action.busy} onClick={() => action.run(() => api.createRemediation(d.id))}>Create remediation from latest investigation</button>}
        {action.error && <p className="error" role="alert">{action.error}</p>}
      </Card>
    );
  }
  return <>{d.remediations.map((r) => <RemediationCard key={r.id} id={r.id} stamp={r.updated_at} onChange={onChange} />)}</>;
}

const STEPS = ["PROPOSED", "VALIDATED", "PENDING_APPROVAL", "APPROVED", "EXECUTED", "COMPLETED"];
const reached = (r: Remediation, step: string) => {
  const order = ["PROPOSED", "VALIDATING", "VALIDATED", "PENDING_APPROVAL", "APPROVED", "EXECUTED", "COMPLETED"];
  if (r.state === "ROLLED_BACK") return true;
  if (r.state === "VALIDATION_FAILED") return step === "PROPOSED";
  if (r.state === "REJECTED") return order.indexOf(step) <= (r.validation?.passed ? order.indexOf("PENDING_APPROVAL") : 0);
  return order.indexOf(step) <= order.indexOf(r.state);
};

function RemediationCard({ id, stamp, onChange }: { id: string; stamp: string; onChange: () => void }) {
  const { data: r, error, loading, reload } = useAsync(() => api.remediation(id), [id, stamp]);
  const action = useAction(() => { void reload(); onChange(); });
  const [comment, setComment] = useState("");
  return (
    <Card title={`Remediation ${id}`} aside={r ? <Status value={r.state} /> : undefined}>
      <StateBox loading={loading && !r} error={error}>
        {r && (
          <>
            <div className={`banner ${r.executed ? "banner-executed" : "banner-proposal"}`}>
              {r.state === "ROLLED_BACK" ? `ROLLED BACK — the deployment was restored to the state of revision #${r.base_revision} (now revision #${r.rollback_revision}).`
                : r.executed ? `EXECUTED — deployment revision #${r.executed_revision} is the result of this action.`
                : "RECOMMENDATION ONLY — nothing has been changed. Execution requires an authenticated approver."}
            </div>
            <ol className="lifecycle">{STEPS.map((s) => <li key={s} className={reached(r, s) ? "done" : ""}>{s.replace(/_/g, " ").toLowerCase()}</li>)}</ol>
            <dl className="kv">
              <dt>Proposed action</dt><dd><b>{r.action.replace(/_/g, " ")}</b> {Object.keys(r.proposal.parameters).length > 0 && <code>{JSON.stringify(r.proposal.parameters)}</code>}</dd>
              <dt>Rationale</dt><dd>{r.proposal.rationale}</dd>
              <dt>Expected effect</dt><dd>{r.proposal.expected_effect}</dd>
              <dt>Risk classification</dt><dd><Severity value={r.risk_level} /></dd>
              <dt>Rollback plan</dt><dd>{r.proposal.rollback_plan}</dd>
              {r.proposal.caveats.length > 0 && <><dt>Caveats</dt><dd>{r.proposal.caveats.join(" ")}</dd></>}
              {r.proposal.runbook_refs.length > 0 && <><dt>Runbooks</dt><dd>{r.proposal.runbook_refs.map((x) => <code key={x}>{x} </code>)}</dd></>}
            </dl>
            <h4>Sandbox validation {r.validation ? <Status value={r.validation.passed ? "ok" : "failed"} /> : <span className="muted">not run</span>}</h4>
            {r.validation && (
              <table><thead><tr><th>Check</th><th>Category</th><th>Result</th><th>Detail</th></tr></thead>
                <tbody>{r.validation.checks.map((c) => (
                  <tr key={c.name}><td>{c.name.replace(/_/g, " ")}</td><td>{c.category}</td><td><Status value={c.passed ? "ok" : "failed"} /></td><td>{c.detail}</td></tr>))}</tbody></table>
            )}
            {r.validation && <p className="muted small">Regression checks: {r.validation.checks.filter((c) => c.category === "regression").every((c) => c.passed) ? "all passed" : "FAILED"} · validated against deployment revision #{r.base_revision} in {fmtMs(r.validation.duration_ms)}</p>}
            {r.policy_decision && (
              <>
                <h4>Policy decision <Status value={r.policy_decision.outcome} /></h4>
                <p className="small">score {r.policy_decision.score.toFixed(2)} = benefit {r.policy_decision.expected_benefit.toFixed(2)} − risk {r.policy_decision.failure_risk.toFixed(2)} − cost {r.policy_decision.execution_cost.toFixed(2)} <span className="muted">({r.policy_decision.policy_version})</span></p>
                <ul className="plain small">{[...r.policy_decision.reasons, ...r.policy_decision.unmet_requirements].map((x, i) => <li key={i}>{x}</li>)}</ul>
              </>
            )}
            <h4>Approval</h4>
            {r.approval ? <p>{r.approval.decision} by <b>{r.approval.reviewer}</b> ({r.approval.reviewer_role}) at {fmtTime(r.approval.decided_at)}{r.approval.comment ? ` — “${r.approval.comment}”` : ""}</p>
              : <p className="muted">{r.state === "PENDING_APPROVAL" ? "Waiting for a human approver." : "No decision recorded."}</p>}
            {r.verification && <p>Post-execution verification: <Status value={r.verification.passed ? "ok" : "failed"} /> {r.verification.summary}</p>}
            <h4>Rollback</h4>
            <p className="muted">{r.rollback_revision ? `Rolled back; deployment revision #${r.rollback_revision} restores the pre-remediation state.` : r.executed ? "Not rolled back." : "Nothing to roll back."}</p>
            {action.error && <p className="error" role="alert">{action.error}</p>}
            <div className="actions">
              {r.state === "PROPOSED" && <button disabled={action.busy} onClick={() => action.run(() => api.validate(r.id))}>Run sandbox validation</button>}
              {r.state === "PENDING_APPROVAL" && (
                <>
                  <input placeholder="Decision comment" value={comment} onChange={(e) => setComment(e.target.value)} maxLength={2000} />
                  <button disabled={action.busy} onClick={() => action.run(() => api.decide(r.id, "approve", r.version, comment))}>Approve and execute</button>
                  <button className="ghost" disabled={action.busy} onClick={() => action.run(() => api.decide(r.id, "reject", r.version, comment))}>Reject</button>
                </>
              )}
              {(r.state === "EXECUTED" || r.state === "COMPLETED") && (
                <button className="ghost" disabled={action.busy || comment.trim().length < 3} title="Enter a reason first"
                  onClick={() => action.run(() => api.rollback(r.id, comment))}>Roll back</button>
              )}
              {(r.state === "EXECUTED" || r.state === "COMPLETED") && <input placeholder="Rollback reason (required)" value={comment} onChange={(e) => setComment(e.target.value)} maxLength={500} />}
            </div>
            <p className="muted small">Approve, reject and roll back require the approver role; the server checks it from your token.</p>
          </>
        )}
      </StateBox>
    </Card>
  );
}

function AuditTab({ id }: { id: string }) {
  const { data, error, loading } = useAsync(() => api.audit(id), [id], 5000);
  return (
    <Card title="Audit trail" aside={<span className="muted">append-only</span>}>
      <StateBox loading={loading && !data} error={error} empty={data?.items.length === 0} emptyText="No audit events.">
        {data && (
          <table><thead><tr><th>Time</th><th>Actor</th><th>Action</th><th>Outcome</th><th>Entity</th><th>Detail</th></tr></thead>
            <tbody>{data.items.map((e) => (
              <tr key={e.id}><td>{fmtTime(e.at)}</td><td>{e.actor} <span className="muted">({e.actor_role})</span></td><td><code>{e.action}</code></td>
                <td><Status value={e.outcome} /></td><td><code>{e.entity_id}</code></td>
                <td><details><summary>{Object.keys(e.detail).length} field(s)</summary><pre>{JSON.stringify(e.detail, null, 1)}</pre></details></td></tr>))}</tbody></table>
        )}
      </StateBox>
    </Card>
  );
}
