import { vi } from "vitest";

type Reply = { status?: number; body?: unknown; text?: string };
type Handler = (url: string, init: RequestInit) => Reply | Promise<never>;

/** Replace global fetch with a router. Unmatched requests fail the test loudly. */
export function mockApi(routes: Record<string, Handler | Reply>) {
  const calls: { method: string; path: string; headers: Record<string, string>; body: any }[] = [];
  const fn = vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = String(input).replace(/^\/api/, "");
    const method = (init.method ?? "GET").toUpperCase();
    const headers = Object.fromEntries(Object.entries((init.headers ?? {}) as Record<string, string>));
    calls.push({ method, path: url, headers, body: init.body ? JSON.parse(String(init.body)) : undefined });
    const key = Object.keys(routes).find((k) => {
      const [m, p] = k.split(" ");
      return m === method && (p.endsWith("*") ? url.startsWith(p.slice(0, -1)) : url.split("?")[0] === p);
    });
    if (!key) throw new Error(`unmocked request: ${method} ${url}`);
    const route = routes[key];
    const result = typeof route === "function" ? await route(url, init) : route;
    const status = result.status ?? 200;
    return {
      ok: status >= 200 && status < 300, status, statusText: `HTTP ${status}`,
      json: async () => { if (result.text !== undefined) throw new SyntaxError("not json"); return result.body; },
    } as Response;
  });
  vi.stubGlobal("fetch", fn);
  return { calls, fn };
}

export const READY = { status: "ready", components: {
  database: { status: "ok", required: true, detail: "" },
  llm: { status: "unavailable", required: false, detail: "ConnectError; deterministic diagnosis remains available" } } };

export const OVERVIEW = {
  incidents: { total: 3, active: 1, critical_active: 0, by_status: {}, by_severity: {} },
  pending_approvals: 1, monitoring_runs: 7,
  datasets: [{ dataset: "adult", deployment: { revision: 2, dataset: "adult", model_version: "adult-hgb-1", quarantined: false, ingestion: {} },
    last_run: { at: "2026-11-12T04:00:00Z", rows: 2000, duration_ms: 812, incident_detected: true, max_severity: "medium",
      drift: { features_tested: 14, features_drifted: ["capital_loss"] },
      performance: { labelled_rows: 2000, reference: { accuracy: 0.87, f1: 0.7 }, current: null } } }],
};

export const INCIDENT = {
  id: "INC-1", dataset: "adult", incident_type: "data_quality", severity: "medium", status: "awaiting_approval",
  model_version: "adult-hgb-1", reference_version: "ref-1", summary: "4.4% of capital_loss values out of range",
  detected_at: "2026-11-12T04:00:00Z", updated_at: "2026-11-12T04:05:00Z", monitoring_run_id: "REP-1", batch_id: "BAT-1",
  findings: [{ finding_id: "FND-1", monitor: "quality", check: "range_violation_rate", metric_name: "rate", feature: "capital_loss",
    observed_value: 0.044, reference_value: 0, threshold: 0.01, severity: "medium", level: "observed_fact",
    interpretation: "4.4% of 'capital_loss' values are outside the reference range" }],
  skipped_checks: [],
  performance: { reference: { accuracy: 0.87, roc_auc: 0.93 }, current: { accuracy: 0.86, roc_auc: 0.9 }, n_labelled: 2000 },
  evidence: [
    { evidence_id: "EV-1", source: "monitoring:quality", level: "observed_fact", severity: "medium", feature: "capital_loss",
      statement: "range violations in capital_loss", finding_id: "FND-1", investigation_id: null, data: { polarity: "anomaly" } },
    { evidence_id: "EV-ok-1", source: "monitoring:summary", level: "observed_fact", severity: "info", feature: null,
      statement: "the batch conforms to the data contract", finding_id: null, investigation_id: null, data: { polarity: "normal" } },
  ],
  investigations: [{ id: "INV-1", status: "completed" }],
  remediations: [{ id: "REM-1", updated_at: "2026-11-12T04:05:00Z" }],
};

export const INVESTIGATION = {
  id: "INV-1", incident_id: "INC-1", status: "completed", llm_status: "disabled", llm_model: null, duration_ms: 1900,
  requested_by: "bob", started_at: "2026-11-12T04:01:00Z", errors: [],
  explanation: { source: "deterministic", summary: "Conclusion: consistent with scaling error.", operator_checks: [] },
  diagnosis: { conclusion: "The evidence is most consistent with scaling error. This is a ranked hypothesis, not established causation." },
  hypotheses: [{ hypothesis_id: "HYP-1", rank: 1, root_cause: "scaling_error", confidence: 0.88, ranked_by: "rules",
    supporting_evidence_ids: ["EV-1"], contradicting_evidence_ids: [],
    payload: { reasoning: "constant-factor change", missing_evidence: [] } }],
  tool_traces: [{ id: 1, seq: 0, tool: "inspect_feature_distribution", requested_by: "rules", status: "ok", duration_ms: 30, error: null, output: { summary: "scale x57" } }],
  llm_calls: [], events: [{ node: "collect_incident", status: "ok", duration_ms: 3 }], runbook_refs: [],
};

export const REMEDIATION = {
  id: "REM-1", incident_id: "INC-1", state: "PENDING_APPROVAL", action: "fix_ingestion_scale", risk_level: "medium",
  version: 4, executed: false, base_revision: 1, executed_revision: null, rollback_revision: null, updated_at: "2026-11-12T04:05:00Z",
  proposal: { parameters: { feature: "capital_loss", factor: 0.0173 }, rationale: "arrives multiplied by ~57.7",
    expected_effect: "range violations clear", rollback_plan: "re-activate revision 1", caveats: [], runbook_refs: ["RB-004#symptoms"] },
  validation: { passed: true, duration_ms: 1100, checks: [
    { name: "targeted_findings_cleared", category: "effect", passed: true, detail: "all cleared" },
    { name: "performance_not_degraded", category: "regression", passed: true, detail: "ROC-AUC 0.898 vs 0.902" }] },
  policy_decision: { outcome: "REQUEST_HUMAN_APPROVAL", score: 0.72, expected_benefit: 0.79, failure_risk: 0.02, execution_cost: 0.05,
    policy_version: "policy-1.0", reasons: ["validated in sandbox; medium-risk state change requires human approval"], unmet_requirements: [] },
  approval: null, verification: null,
};

const rate = (k: number, n: number, definition = "k / n") => ({ value: n ? k / n : null, k, n, ci95: [0.1, 0.9] as [number, number], definition });
const detection = (p: string, pos: number, tp: number, f1: number, faults: string[]) => ({
  metric_prefix: p, definition: `definition of ${p}`, positive_condition: "cond",
  population: { included_fault_categories: faults, n_positive_scenarios: pos, n_clean_scenarios: 50, n_scenarios: pos + 50 },
  confusion: { tp, fp: 0, tn: 50, fn: pos - tp },
  metrics: { [`${p}_precision`]: rate(tp, tp), [`${p}_recall`]: rate(tp, pos), [`${p}_false_positive_rate`]: rate(0, 50),
    [`${p}_f1`]: { value: f1, ci95: [0.9, 1.0], ci_method: "bootstrap", definition: "f1" } },
});
const config = (extra: object = {}) => ({ scenarios_diagnosed: 119, root_cause_top1: rate(115, 119), root_cause_recall_at_3: rate(116, 119),
  claim_validation: null, remediation: null, ...extra });
export const EVAL_RUN = {
  name: "main", n_scenarios: 170, seeds: [201, 202], llm_model: "qwen3:4b", methodology: ["note one"],
  detection: [detection("overall_fault_detection", 120, 119, 0.9958, ["F01", "F02", "F03"]), detection("drift_detection", 36, 33, 0.9565, ["F01", "F02", "F04"])],
  targets: { drift_detection_f1: { target: ">= 0.9", observed: 0.9565, met: true } },
  full_grid: { A: config(), B: config({ remediation: {
    counts: { investigations: 119, proposals_total: 119, proposals_no_state_change: 32, proposals_actionable: 87,
      rejected_by_policy_before_validation: 2, validation_attempts: 85, validation_failed: 17, approval_requests: 68, executed_fixes: 68, rollback_attempts: 68 },
    simulation_note: "simulated in-process",
    rates: { validation_success_rate: rate(68, 85, "validation attempts that passed / all validation attempts") },
    lifecycle: { reconciled: true, events: 748, proposals: 119, derived_events: 748, terminal_status_counts: {}, failure_counts: {} } } }) },
  llm_subset: { n: 20, configs: { A: config(), B: config(), C: config(),
    D: config({ claim_validation: { available: false, reason: "this run recorded no claim-validation counters" } }) } },
};
