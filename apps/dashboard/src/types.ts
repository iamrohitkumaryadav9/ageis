// Mirrors the API's response models (apps/api/schemas.py).
export type Json = Record<string, any>;

export interface Page<T> { items: T[]; total: number; limit: number; offset: number }

export interface Incident {
  id: string; dataset: string; incident_type: string; severity: string; status: string;
  model_version: string; reference_version: string; summary: string; detected_at: string;
  updated_at: string; monitoring_run_id: string; batch_id: string;
}
export interface Evidence {
  evidence_id: string; source: string; level: string; severity: string; feature: string | null;
  statement: string; finding_id: string | null; investigation_id: string | null; data: Json;
}
export interface Finding {
  finding_id: string; monitor: string; check: string; metric_name: string; feature: string | null;
  observed_value: number | string | boolean | null; reference_value: number | string | boolean | null;
  threshold: number | string | boolean | null; severity: string; level: string; interpretation: string;
}
export interface InvestigationSummary {
  id: string; incident_id: string; status: string; requested_by: string; use_llm: boolean;
  llm_status: string | null; llm_model: string | null; started_at: string; finished_at: string | null;
  duration_ms: number | null;
}
export interface ValidationCheck { name: string; passed: boolean; detail: string; category: string }
export interface Remediation {
  id: string; incident_id: string; state: string; action: string; risk_level: string; version: number;
  proposal: { rationale: string; parameters: Json; expected_effect: string; rollback_plan: string;
    caveats: string[]; runbook_refs: string[]; evidence_ids: string[]; target_root_cause: string | null };
  validation: { passed: boolean; summary: string; checks: ValidationCheck[]; metrics_before: Json;
    metrics_after: Json; duration_ms: number } | null;
  policy_decision: { outcome: string; score: number; expected_benefit: number; failure_risk: number;
    execution_cost: number; reasons: string[]; unmet_requirements: string[]; policy_version: string;
    requires_approval: boolean } | null;
  verification: { passed: boolean; summary: string } | null;
  base_revision: number; executed_revision: number | null; rollback_revision: number | null;
  created_by: string; created_at: string; updated_at: string;
  approval?: { decision: string; reviewer: string; reviewer_role: string; comment: string; decided_at: string } | null;
  allowed_events?: string[]; executed?: boolean;
}
export interface IncidentDetail extends Incident {
  findings: Finding[]; skipped_checks: { monitor: string; check: string; feature: string | null; reason: string }[];
  performance: { reference: Json | null; current: Json | null; n_labelled: number };
  evidence: Evidence[]; investigations: InvestigationSummary[]; remediations: Remediation[];
}
export interface Hypothesis {
  rank: number; hypothesis_id: string; root_cause: string; confidence: number; ranked_by: string;
  supporting_evidence_ids: string[]; contradicting_evidence_ids: string[];
  payload: { reasoning: string; missing_evidence: string[]; affected_features: string[]; next_diagnostic_action: string };
}
export interface ToolTrace {
  id: string; seq: number; tool: string; requested_by: string; status: string; requested_args: Json;
  validated_args: Json | null; output: { summary: string; data: Json } | null; error: string | null;
  duration_ms: number; started_at: string;
}
export interface InvestigationDetail extends InvestigationSummary {
  diagnosis: { conclusion: string; sufficient_evidence: boolean; engine: string } | null;
  recommendation: Json | null; policy_decision: Json | null;
  explanation: { source: string; summary: string; operator_checks: string[]; review_flags?: string[] } | null;
  runbook_refs: { passage_id: string; doc_title: string; doc_version: string; section: string; text: string;
    score: number; suspicious: boolean }[];
  llm_calls: { purpose: string; ok: boolean; model: string | null; duration_ms: number; error: string | null;
    prompt_tokens: number | null; completion_tokens: number | null; attempts: number }[];
  events: { node: string; status: string; duration_ms: number }[];
  errors: string[]; trace_id: string | null; tool_traces: ToolTrace[]; hypotheses: Hypothesis[];
  llm_ranking_stats: Json | null;
}
export interface Overview {
  incidents: { total: number; active: number; critical_active: number; by_status: Record<string, number>;
    by_severity: Record<string, number> };
  pending_approvals: number; monitoring_runs: number;
  datasets: {
    dataset: string;
    deployment: (Json & { revision: number; model_version: string; quarantined: boolean }) | null;
    last_run: { id: string; at: string; incident_detected: boolean; max_severity: string; rows: number;
      duration_ms: number; drift: { features_tested: number; features_drifted: string[] };
      performance: { current: Record<string, number | null> | null; reference: Record<string, number | null> | null;
        labelled_rows: number } } | null;
  }[];
}
export interface Ready { status: string; components: Record<string, { status: string; required: boolean; detail: string }> }
export interface AuditEvent {
  id: number; at: string; actor: string; actor_role: string; action: string; entity_type: string;
  entity_id: string; outcome: string; detail: Json;
}

// ---- benchmark results (GET /evaluation/runs/{name})
export interface EvalRate { value: number | null; k: number; n: number; ci95: [number, number]; definition?: string }
export interface DetectionMetric {
  metric_prefix: string; definition: string; positive_condition: string;
  population: { included_fault_categories: string[]; n_positive_scenarios: number; n_clean_scenarios: number; n_scenarios: number };
  confusion: { tp: number; fp: number; tn: number; fn: number };
  metrics: Record<string, EvalRate>;
}
export interface ClaimValidation {
  available: boolean; reason?: string; counters?: Record<string, number>;
  rates?: Record<string, EvalRate>; limits?: string;
}
export interface EvalConfig {
  scenarios_diagnosed: number; root_cause_top1: EvalRate; root_cause_recall_at_3: EvalRate;
  claim_validation: ClaimValidation | null;
  remediation: null | {
    counts: Record<string, number>; simulation_note: string; rates: Record<string, EvalRate>;
    lifecycle: { reconciled: boolean; events: number; proposals: number; derived_events: number;
      terminal_status_counts: Record<string, number>; failure_counts: Record<string, number> };
  };
}
export interface EvalRun {
  name: string; n_scenarios: number; seeds: number[] | null; llm_model: string | null; methodology: string[];
  detection: DetectionMetric[];
  targets: Record<string, { target: string; observed: number | null; met: boolean }>;
  full_grid: Record<"A" | "B", EvalConfig | null>;
  llm_subset: { n: number; configs: Record<"A" | "B" | "C" | "D", EvalConfig | null> };
}
