import { useState } from "react";
import { api } from "./api";
import type { ClaimValidation, DetectionMetric, EvalConfig, EvalRate, EvalRun } from "./types";
import { Card, StateBox, Status, useAsync } from "./ui";

/** A rate is never shown without its numerator, denominator and interval. */
export function Rate({ rate }: { rate: EvalRate | null | undefined }) {
  if (!rate || rate.value == null) return <span className="muted">not available{rate ? ` (0 of ${rate.n})` : ""}</span>;
  return (
    <span className="rate">
      <b>{rate.value.toFixed(3)}</b> ({rate.k}/{rate.n}, 95% CI {rate.ci95[0].toFixed(2)}–{rate.ci95[1].toFixed(2)})
    </span>
  );
}

const TITLES: Record<string, string> = {
  overall_fault_detection: "Overall fault detection",
  drift_detection: "Drift detection",
};

export function DetectionCard({ metric }: { metric: DetectionMetric }) {
  const p = metric.metric_prefix;
  const f1 = metric.metrics[`${p}_f1`] as unknown as { value: number; ci95: [number, number] };
  return (
    <Card title={TITLES[p] ?? p} aside={<code>{p}</code>}>
      <p className="small">{metric.definition}</p>
      <dl className="kv compact">
        <dt>Fault categories</dt><dd>{metric.population.included_fault_categories.join(", ")}</dd>
        <dt>Positive scenarios</dt><dd data-testid={`${p}-positives`}>{metric.population.n_positive_scenarios}</dd>
        <dt>Clean scenarios</dt><dd>{metric.population.n_clean_scenarios}</dd>
        <dt>TP / FP / TN / FN</dt>
        <dd>{metric.confusion.tp} / {metric.confusion.fp} / {metric.confusion.tn} / {metric.confusion.fn}</dd>
        <dt><code>{p}_f1</code></dt>
        <dd data-testid={`${p}_f1`}><b>{f1.value.toFixed(4)}</b> (95% CI {f1.ci95[0].toFixed(3)}–{f1.ci95[1].toFixed(3)}, bootstrap)</dd>
        <dt><code>{p}_precision</code></dt><dd><Rate rate={metric.metrics[`${p}_precision`]} /></dd>
        <dt><code>{p}_recall</code></dt><dd><Rate rate={metric.metrics[`${p}_recall`]} /></dd>
        <dt><code>{p}_false_positive_rate</code></dt><dd><Rate rate={metric.metrics[`${p}_false_positive_rate`]} /></dd>
      </dl>
    </Card>
  );
}

function Remediation({ config, label }: { config: EvalConfig; label: string }) {
  const rem = config.remediation;
  if (!rem) return null;
  const c = rem.counts;
  const rows: [string, number, string][] = [
    ["Investigations", c.investigations, ""],
    ["No state change proposed", c.proposals_no_state_change, "deferred; never validated"],
    ["Actionable proposals", c.proposals_actionable, `${c.proposals_total} − ${c.proposals_no_state_change}`],
    ["Rejected by policy before validation", c.rejected_by_policy_before_validation, "never validated"],
    ["Validation attempts", c.validation_attempts, `${c.proposals_actionable} − ${c.rejected_by_policy_before_validation}`],
    ["Validation failed", c.validation_failed, ""],
    ["Approval requests", c.approval_requests, `${c.validation_attempts} − ${c.validation_failed}`],
    ["Executed fixes", c.executed_fixes, ""],
    ["Rollback attempts", c.rollback_attempts, ""],
  ];
  return (
    <Card title={`Remediation lifecycle — ${label}`}
      aside={<span data-testid="reconciled"><Status value={rem.lifecycle.reconciled ? "ok" : "failed"} /> {rem.lifecycle.reconciled ? "reconciled" : "reconciliation failed"}</span>}>
      <p className="small muted">{rem.simulation_note}</p>
      <table><thead><tr><th>Stage</th><th>Count</th><th>From the stage above</th></tr></thead>
        <tbody>{rows.map(([name, n, how]) => <tr key={name}><td>{name}</td><td>{n}</td><td className="muted">{how}</td></tr>)}</tbody></table>
      <h4>Rates, with their denominators</h4>
      <table><thead><tr><th>Rate</th><th>Value</th><th>Definition</th></tr></thead>
        <tbody>{Object.entries(rem.rates).map(([name, r]) => (
          <tr key={name}><td><code>{name}</code></td><td><Rate rate={r} /></td><td className="small">{r.definition}</td></tr>
        ))}</tbody></table>
      {!rem.lifecycle.reconciled && (
        <ul className="notes">{Object.entries(rem.lifecycle.failure_counts).map(([code, n]) => <li key={code}>{code}: {n}</li>)}</ul>
      )}
      {rem.lifecycle.derived_events > 0 &&
        <p className="small muted">{rem.lifecycle.derived_events} of {rem.lifecycle.events} lifecycle events were derived from summary fields of an older run, not recorded live.</p>}
    </Card>
  );
}

export function Claims({ claims, label }: { claims: ClaimValidation | null | undefined; label: string }) {
  if (!claims) return null;
  if (!claims.available) {
    return <Card title={`Claim validation — ${label}`}><p className="muted" data-testid="claims-unavailable">Unavailable: {claims.reason}.</p></Card>;
  }
  const c = claims.counters ?? {};
  const keys = Object.keys(c);
  return (
    <Card title={`Claim validation — ${label}`}>
      <table><thead><tr><th>Counter</th><th>Count</th></tr></thead>
        <tbody>{keys.map((k) => <tr key={k}><td><code>{k}</code></td><td>{c[k]}</td></tr>)}</tbody></table>
      <h4>Rates</h4>
      <table><thead><tr><th>Rate</th><th>Value</th><th>Definition</th></tr></thead>
        <tbody>{Object.entries(claims.rates ?? {}).map(([name, r]) => (
          <tr key={name}><td><code>{name}</code></td><td><Rate rate={r} /></td><td className="small">{r.definition}</td></tr>
        ))}</tbody></table>
      <p className="small muted">{claims.limits}</p>
    </Card>
  );
}

function RunView({ run }: { run: EvalRun }) {
  const d = run.llm_subset.configs.D;
  return (
    <>
      <p className="muted small">
        Run <code>{run.name}</code> · {run.n_scenarios} scenarios · evaluation seeds {(run.seeds ?? []).join(", ")}
        {run.llm_subset.n > 0 ? ` · LLM configurations on ${run.llm_subset.n} scenarios${run.llm_model ? ` (${run.llm_model})` : ""}` : " · deterministic configurations only"}
      </p>
      <div className="state" role="note">
        Two detection metrics are reported. They cover different scenario populations and define a detection differently, so their values are not interchangeable.
      </div>
      <div className="grid-2">{run.detection.map((m) => <DetectionCard key={m.metric_prefix} metric={m} />)}</div>
      <Card title="Engineering targets">
        <table><thead><tr><th>Metric</th><th>Goal</th><th>Observed</th><th>Met</th></tr></thead>
          <tbody>{Object.entries(run.targets).map(([name, t]) => (
            <tr key={name}><td><code>{name}</code></td><td>{t.target}</td><td>{String(t.observed)}</td>
              <td><Status value={t.met ? "ok" : "failed"} /></td></tr>
          ))}</tbody></table>
      </Card>
      <Card title="Root-cause diagnosis">
        <table><thead><tr><th>Configuration</th><th>Scope</th><th>Top-1</th><th>Recall@3</th></tr></thead>
          <tbody>
            {(["A", "B"] as const).map((c) => run.full_grid[c] && (
              <tr key={`full-${c}`}><td>{c}</td><td>full grid</td>
                <td><Rate rate={run.full_grid[c]!.root_cause_top1} /></td><td><Rate rate={run.full_grid[c]!.root_cause_recall_at_3} /></td></tr>
            ))}
            {(["A", "B", "C", "D"] as const).map((c) => run.llm_subset.configs[c] && (
              <tr key={`sub-${c}`}><td>{c}</td><td>LLM subset (n={run.llm_subset.n})</td>
                <td><Rate rate={run.llm_subset.configs[c]!.root_cause_top1} /></td><td><Rate rate={run.llm_subset.configs[c]!.root_cause_recall_at_3} /></td></tr>
            ))}
          </tbody></table>
      </Card>
      {run.full_grid.B && <Remediation config={run.full_grid.B} label="configuration B, full grid" />}
      {d && <Claims claims={d.claim_validation} label="configuration D (full agent)" />}
      {run.methodology.length > 0 && (
        <Card title="Methodology notes"><ul>{run.methodology.map((m) => <li key={m}>{m}</li>)}</ul></Card>
      )}
    </>
  );
}

export function EvaluationPage() {
  const runs = useAsync(() => api.evaluationRuns(), []);
  const [chosen, setChosen] = useState("");
  // Default to the reference run when it exists, otherwise the most recent name.
  const available = runs.data?.runs ?? [];
  const name = chosen || (available.includes("main") ? "main" : available[available.length - 1]) || "";
  const run = useAsync(() => (name ? api.evaluationRun(name) : Promise.resolve(null)), [name]);
  return (
    <StateBox loading={runs.loading} error={runs.error} empty={runs.data?.runs.length === 0}
      emptyText="No benchmark results are available on this server. Generate them with: python -m aegis.evaluation.run, then python -m aegis.evaluation.report.">
      <div className="page-head">
        <h2>Benchmark results (AEGIS-IR)</h2>
        <label className="filter">Run
          <select aria-label="Benchmark run" value={name} onChange={(e) => setChosen(e.target.value)}>
            {(runs.data?.runs ?? []).map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
        </label>
      </div>
      <StateBox loading={run.loading} error={run.error}>{run.data && <RunView run={run.data} />}</StateBox>
    </StateBox>
  );
}
