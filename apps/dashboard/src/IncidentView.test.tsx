import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { IncidentView } from "./IncidentView";
import { INCIDENT, INVESTIGATION, mockApi, REMEDIATION } from "./test/fixtures";

beforeEach(() => { sessionStorage.setItem("aegis.token", "t"); });
afterEach(cleanup);

const base = (extra: Record<string, any> = {}) => ({
  "GET /incidents/INC-1": { body: INCIDENT },
  "GET /investigations/INV-1": { body: INVESTIGATION },
  "GET /remediation/REM-1": { body: REMEDIATION },
  ...extra,
});
const open = async (tab?: string) => {
  render(<IncidentView id="INC-1" />);
  await waitFor(() => expect(screen.getByText("4.4% of capital_loss values out of range")).toBeTruthy());
  if (tab) await userEvent.click(screen.getByRole("tab", { name: tab }));
};

describe("incident detail", () => {
  it("renders findings and evidence, labelled as measurements rather than causes", async () => {
    mockApi(base());
    await open();
    expect(screen.getByText("what was measured — not what caused it")).toBeTruthy();
    expect(screen.getByText("quality · range_violation_rate")).toBeTruthy();
    expect(screen.getByText("Evidence records (2)")).toBeTruthy();
    expect(screen.getByText("EV-1")).toBeTruthy();
    expect(screen.getByText("normal")).toBeTruthy();  // normal evidence is shown as such
  });

  it("shows a not-found error from the API", async () => {
    mockApi({ "GET /incidents/INC-1": { status: 404, body: { error: { code: "not_found", message: "incident 'INC-1' not found" } } } });
    render(<IncidentView id="INC-1" />);
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("incident 'INC-1' not found"));
  });

  it("renders the diagnosis as ranked hypotheses with their evidence", async () => {
    mockApi(base());
    await open("Investigation");
    await waitFor(() => expect(screen.getByText("scaling error")).toBeTruthy());
    expect(screen.getByText("ranked by evidence fit — hypotheses, not established causes")).toBeTruthy();
    expect(screen.getByText(/not established causation/)).toBeTruthy();
    expect(screen.getByText("fit 0.88")).toBeTruthy();
    expect(screen.getByText("Supporting evidence (1)")).toBeTruthy();
    expect(screen.getByText("Contradicting evidence: none")).toBeTruthy();
    expect(screen.getByText(/No LLM requests were made/)).toBeTruthy();
  });

  it("starts an investigation with the chosen LLM setting", async () => {
    const { calls } = mockApi(base({ "POST /incidents/INC-1/investigate": { body: { id: "INV-2", status: "running" } } }));
    await open("Investigation");
    await userEvent.click(screen.getByRole("checkbox"));
    await userEvent.click(screen.getByRole("button", { name: "Investigate again" }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST")).toBe(true));
    expect(calls.find((c) => c.method === "POST")!.body).toEqual({ use_llm: false });
  });
});

describe("remediation and approval", () => {
  it("presents a pending proposal as a recommendation, with validation and policy", async () => {
    mockApi(base());
    await open("Remediation");
    await waitFor(() => expect(screen.getByText(/RECOMMENDATION ONLY — nothing has been changed/)).toBeTruthy());
    expect(screen.getByText("fix ingestion scale")).toBeTruthy();
    expect(screen.getByText("targeted findings cleared")).toBeTruthy();
    expect(screen.getByText(/score 0\.72 = benefit 0\.79 − risk 0\.02 − cost 0\.05/)).toBeTruthy();
    expect(screen.getByText("Waiting for a human approver.")).toBeTruthy();
    expect(screen.getByText("Nothing to roll back.")).toBeTruthy();
    const done = [...document.querySelectorAll("ol.lifecycle li.done")].map((li) => li.textContent);
    expect(done).toEqual(["proposed", "validated", "pending approval"]);
    expect(screen.queryByRole("button", { name: "Roll back" })).toBeNull();
  });

  it("approving sends the version the reviewer saw and the comment", async () => {
    const { calls } = mockApi(base({ "POST /remediation/REM-1/approve": { body: { ...REMEDIATION, state: "COMPLETED", executed: true } } }));
    await open("Remediation");
    await waitFor(() => expect(screen.getByRole("button", { name: "Approve and execute" })).toBeTruthy());
    await userEvent.type(screen.getByPlaceholderText("Decision comment"), "verified with owner");
    await userEvent.click(screen.getByRole("button", { name: "Approve and execute" }));
    await waitFor(() => expect(calls.some((c) => c.path === "/remediation/REM-1/approve")).toBe(true));
    const call = calls.find((c) => c.path === "/remediation/REM-1/approve")!;
    expect(call.body).toEqual({ expected_version: 4, comment: "verified with owner" });
    expect(call.headers.Authorization).toBe("Bearer t");
  });

  it("shows the server's refusal when a non-approver tries to approve", async () => {
    mockApi(base({ "POST /remediation/REM-1/approve": { status: 403,
      body: { error: { code: "forbidden", message: "the approver role is required to decide a remediation" } } } }));
    await open("Remediation");
    await waitFor(() => expect(screen.getByRole("button", { name: "Approve and execute" })).toBeTruthy());
    await userEvent.click(screen.getByRole("button", { name: "Approve and execute" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("the approver role is required"));
    expect(screen.getByRole("alert").textContent).toContain("(403)");
    expect(screen.getByText(/RECOMMENDATION ONLY/)).toBeTruthy();  // nothing is shown as executed
  });

  it("shows a stale-version conflict instead of pretending the decision went through", async () => {
    mockApi(base({ "POST /remediation/REM-1/reject": { status: 409,
      body: { error: { code: "conflict", message: "the remediation changed since you loaded it" } } } }));
    await open("Remediation");
    await waitFor(() => expect(screen.getByRole("button", { name: "Reject" })).toBeTruthy());
    await userEvent.click(screen.getByRole("button", { name: "Reject" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("changed since you loaded it (409)"));
  });

  it("an executed remediation is labelled executed and needs a reason to roll back", async () => {
    const executed = { ...REMEDIATION, state: "COMPLETED", executed: true, executed_revision: 2,
      approval: { decision: "approved", reviewer: "alice", reviewer_role: "approver", decided_at: "2026-11-12T04:06:00Z", comment: "ok" },
      verification: { passed: true, summary: "incident batch is healthy" } };
    const { calls } = mockApi(base({ "GET /remediation/REM-1": { body: executed }, "POST /remediation/REM-1/rollback": { body: executed } }));
    await open("Remediation");
    await waitFor(() => expect(screen.getByText(/EXECUTED — deployment revision #2/)).toBeTruthy());
    expect(screen.getByText(/approved by/).textContent).toContain("alice");
    const rollback = screen.getByRole("button", { name: "Roll back" }) as HTMLButtonElement;
    expect(rollback.disabled).toBe(true);
    await userEvent.type(screen.getByPlaceholderText("Rollback reason (required)"), "fix was wrong");
    expect(rollback.disabled).toBe(false);
    await userEvent.click(rollback);
    await waitFor(() => expect(calls.find((c) => c.path.endsWith("/rollback"))?.body).toEqual({ reason: "fix was wrong" }));
    expect(screen.queryByRole("button", { name: "Approve and execute" })).toBeNull();
  });

  it("a failed validation shows no approval controls", async () => {
    const failed = { ...REMEDIATION, state: "VALIDATION_FAILED",
      validation: { ...REMEDIATION.validation, passed: false, checks: [{ name: "performance_not_degraded", category: "regression", passed: false, detail: "AUC fell" }] } };
    mockApi(base({ "GET /remediation/REM-1": { body: failed } }));
    await open("Remediation");
    await waitFor(() => expect(screen.getByText(/Regression checks: FAILED/)).toBeTruthy());
    expect(screen.queryByRole("button", { name: "Approve and execute" })).toBeNull();
    expect(screen.getByText("No decision recorded.")).toBeTruthy();
  });

  it("explains when there is nothing to remediate", async () => {
    mockApi(base({ "GET /incidents/INC-1": { body: { ...INCIDENT, remediations: [] } } }));
    await open("Remediation");
    expect(screen.getByText(/No state-changing remediation is on record/)).toBeTruthy();
  });

  it("renders the audit trail including denied attempts", async () => {
    mockApi(base({ "GET /audit": { body: { items: [
      { id: 1, at: "2026-11-12T04:05:00Z", actor: "bob", actor_role: "operator", action: "remediation.approve", outcome: "denied", entity_id: "REM-1", detail: { reason: "role" } },
      { id: 2, at: "2026-11-12T04:06:00Z", actor: "alice", actor_role: "approver", action: "remediation.approve", outcome: "ok", entity_id: "REM-1", detail: {} },
    ], total: 2, limit: 200, offset: 0 } } }));
    await open("Audit trail");
    await waitFor(() => expect(screen.getAllByText("remediation.approve").length).toBe(2));
    const rows = screen.getAllByRole("row");
    expect(within(rows[1]).getByText("denied")).toBeTruthy();
    expect(within(rows[2]).getByText("ok")).toBeTruthy();
  });
});
