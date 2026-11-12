import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Claims, EvaluationPage, Rate } from "./Evaluation";
import { EVAL_RUN, mockApi } from "./test/fixtures";

beforeEach(() => { sessionStorage.setItem("aegis.token", "t"); });
afterEach(cleanup);
const routes = (run: unknown = EVAL_RUN) => ({ "GET /evaluation/runs": { body: { runs: ["main"] } }, "GET /evaluation/runs/main": { body: run } });

describe("evaluation page", () => {
  it("shows each detection F1 under its own name with its own population", async () => {
    mockApi(routes());
    render(<EvaluationPage />);
    await waitFor(() => expect(screen.getByTestId("overall_fault_detection_f1")).toBeTruthy());
    const overall = screen.getByTestId("overall_fault_detection_f1").textContent!;
    const drift = screen.getByTestId("drift_detection_f1").textContent!;
    expect(overall).toContain("0.9958");
    expect(overall).not.toContain("0.9565");
    expect(drift).toContain("0.9565");
    expect(drift).not.toContain("0.9958");
    expect(screen.getByTestId("overall_fault_detection-positives").textContent).toBe("120");
    expect(screen.getByTestId("drift_detection-positives").textContent).toBe("36");
    expect(screen.getByText(/their values are not interchangeable/)).toBeTruthy();
    expect(screen.getByText("Overall fault detection")).toBeTruthy();
    expect(screen.getByText("Drift detection")).toBeTruthy();
  });

  it("never renders a bare 'F1' label for a detection score", async () => {
    mockApi(routes());
    render(<EvaluationPage />);
    await waitFor(() => expect(screen.getByTestId("drift_detection_f1")).toBeTruthy());
    const labels = [...document.querySelectorAll("dt")].map((dt) => dt.textContent!.trim());
    expect(labels.filter((l) => /f1/i.test(l))).toEqual(["overall_fault_detection_f1", "drift_detection_f1"]);
  });

  it("shows remediation counts, the reconciliation status and rate denominators", async () => {
    mockApi(routes());
    render(<EvaluationPage />);
    await waitFor(() => expect(screen.getByTestId("reconciled")).toBeTruthy());
    expect(screen.getByTestId("reconciled").textContent).toContain("reconciled");
    expect(screen.getByText("Rejected by policy before validation")).toBeTruthy();
    expect(screen.getByText("87 − 2")).toBeTruthy();
    expect(screen.getByText("validation attempts that passed / all validation attempts")).toBeTruthy();
    expect(screen.getByText(/\(68\/85, 95% CI/)).toBeTruthy();
    expect(screen.getByText(/748 of 748 lifecycle events were derived/)).toBeTruthy();
  });

  it("flags a failed reconciliation instead of hiding it", async () => {
    const broken = structuredClone(EVAL_RUN) as any;
    broken.full_grid.B.remediation.lifecycle = { ...broken.full_grid.B.remediation.lifecycle, reconciled: false, failure_counts: { missing_link: 2 } };
    mockApi(routes(broken));
    render(<EvaluationPage />);
    await waitFor(() => expect(screen.getByTestId("reconciled").textContent).toContain("reconciliation failed"));
    expect(screen.getByText("missing_link: 2")).toBeTruthy();
  });

  it("says claim metrics are unavailable rather than showing zeros", async () => {
    mockApi(routes());
    render(<EvaluationPage />);
    await waitFor(() => expect(screen.getByTestId("claims-unavailable")).toBeTruthy());
    expect(screen.getByTestId("claims-unavailable").textContent).toContain("recorded no claim-validation counters");
    expect(screen.queryByText("causal_claims_generated")).toBeNull();
  });

  it("explains an empty server and an API failure", async () => {
    mockApi({ "GET /evaluation/runs": { body: { runs: [] } } });
    const { unmount } = render(<EvaluationPage />);
    await waitFor(() => expect(screen.getByText(/No benchmark results are available/)).toBeTruthy());
    unmount();
    mockApi({ "GET /evaluation/runs": { body: { runs: ["main"] } },
      "GET /evaluation/runs/main": { status: 409, body: { error: { code: "conflict", message: "regenerate it" } } } });
    render(<EvaluationPage />);
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("regenerate it"));
  });
});

describe("Rate and Claims", () => {
  it("renders value, numerator, denominator and interval together", () => {
    render(<Rate rate={{ value: 0.8, k: 68, n: 85, ci95: [0.7, 0.87] }} />);
    expect(document.body.textContent).toContain("0.800 (68/85, 95% CI 0.70–0.87)");
  });

  it("renders an empty denominator as not available, never as 0 or 1", () => {
    render(<Rate rate={{ value: null, k: 0, n: 0, ci95: [0, 1] }} />);
    expect(document.body.textContent).toBe("not available (0 of 0)");
  });

  it("lists generated, rejected and escaped counters separately when available", () => {
    render(<Claims label="D" claims={{ available: true, limits: "lexicon limits",
      counters: { causal_claims_generated: 3, causal_claims_rejected: 2, causal_claims_escalated: 1, causal_claims_escaped: 0 },
      rates: { causal_claims_escaped_rate: { value: 0, k: 0, n: 3, ci95: [0, 0.56], definition: "escaped / generated" } } }} />);
    for (const key of ["causal_claims_generated", "causal_claims_rejected", "causal_claims_escalated", "causal_claims_escaped"])
      expect(screen.getByText(key)).toBeTruthy();
    expect(screen.getByText("escaped / generated")).toBeTruthy();
    expect(screen.getByText("lexicon limits")).toBeTruthy();
  });
});
