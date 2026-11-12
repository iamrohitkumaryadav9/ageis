import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ApiError } from "./api";
import { MetricBars, StateBox, Status, Severity, fmtMs, fmtNum, fmtTime, useAsync } from "./ui";

describe("StateBox", () => {
  it("shows loading, then children", () => {
    const { rerender } = render(<StateBox loading error={null}>content</StateBox>);
    expect(screen.getByText("Loading…")).toBeTruthy();
    rerender(<StateBox loading={false} error={null}>content</StateBox>);
    expect(screen.getByText("content")).toBeTruthy();
  });

  it("shows an empty state with its own text instead of an empty table", () => {
    render(<StateBox loading={false} error={null} empty emptyText="No incidents have been detected.">rows</StateBox>);
    expect(screen.getByText("No incidents have been detected.")).toBeTruthy();
    expect(screen.queryByText("rows")).toBeNull();
  });

  it.each([
    [401, "Enter a valid API token"], [403, "Your role does not allow this."], [0, "Is the API running?"],
  ])("explains error %s to the user", (status, hint) => {
    render(<StateBox loading={false} error={new ApiError(status as number, "x", "boom", "req-7")}>content</StateBox>);
    const alert = screen.getByRole("alert");
    expect(alert.textContent).toContain("boom");
    expect(alert.textContent).toContain(hint as string);
    expect(alert.textContent).toContain("req-7");
    expect(screen.queryByText("content")).toBeNull();
  });

  it("an error takes precedence over loading and never renders stale children", () => {
    render(<StateBox loading error={new ApiError(500, "x", "server error")}>content</StateBox>);
    expect(screen.getByRole("alert").textContent).toContain("Error 500");
  });
});

describe("badges and formatting", () => {
  it("status and severity always carry a text label, not colour alone", () => {
    render(<><Status value="PENDING_APPROVAL" /><Status value={null} /><Severity value="critical" /></>);
    expect(screen.getByText("pending approval")).toBeTruthy();
    expect(screen.getByText("unknown")).toBeTruthy();
    expect(screen.getByText("critical")).toBeTruthy();
  });

  it("formats numbers without inventing values for missing ones", () => {
    expect(fmtNum(null)).toBe("—");
    expect(fmtNum(0.123456)).toBe("0.123");
    expect(fmtNum(7)).toBe("7");
    expect(fmtMs(null)).toBe("—");
    expect(fmtMs(1500)).toBe("1.5 s");
    expect(fmtMs(3.21)).toBe("3.2 ms");
    expect(fmtTime(null)).toBe("—");
  });

  it("says 'not measurable' when the current metric is missing", () => {
    render(<MetricBars rows={[{ name: "f1", reference: 0.7, current: null }]} />);
    expect(screen.getByText(/0\.700 → not measurable/)).toBeTruthy();
  });
});

function Probe({ load }: { load: () => Promise<string> }) {
  const { data, error, loading } = useAsync(load, []);
  return <div>{loading ? "loading" : error ? `error:${error.status}:${error.message}` : `data:${data}`}</div>;
}

describe("useAsync", () => {
  it("moves from loading to data", async () => {
    render(<Probe load={async () => "ok"} />);
    expect(screen.getByText("loading")).toBeTruthy();
    await waitFor(() => expect(screen.getByText("data:ok")).toBeTruthy());
  });

  it("keeps ApiError details and wraps unknown failures", async () => {
    const { unmount } = render(<Probe load={async () => { throw new ApiError(403, "forbidden", "nope"); }} />);
    await waitFor(() => expect(screen.getByText("error:403:nope")).toBeTruthy());
    unmount();
    render(<Probe load={async () => { throw new Error("weird"); }} />);
    await waitFor(() => expect(screen.getByText(/error:0:.*weird/)).toBeTruthy());
  });

  it("polls when asked and stops on unmount", async () => {
    vi.useFakeTimers();
    const load = vi.fn(async () => "x");
    function Poller() { useAsync(load, [], 1000); return null; }
    const { unmount } = render(<Poller />);
    await vi.advanceTimersByTimeAsync(3100);
    expect(load.mock.calls.length).toBeGreaterThanOrEqual(3);
    unmount();
    const after = load.mock.calls.length;
    await vi.advanceTimersByTimeAsync(3000);
    expect(load.mock.calls.length).toBe(after);
    vi.useRealTimers();
  });
});
