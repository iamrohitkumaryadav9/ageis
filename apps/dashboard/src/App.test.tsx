import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { App } from "./App";
import { EVAL_RUN, INCIDENT, mockApi, OVERVIEW, READY } from "./test/fixtures";

const page = (items: unknown[], total = items.length) => ({ items, total, limit: 15, offset: 0 });
beforeEach(() => { sessionStorage.clear(); window.location.hash = "#/"; });
afterEach(cleanup);

describe("App shell", () => {
  it("asks for a token and loads no protected data without one", async () => {
    const { calls } = mockApi({ "GET /ready": { body: READY } });
    render(<App />);
    expect(screen.getByText(/Enter an API token to load data/)).toBeTruthy();
    await waitFor(() => expect(screen.getByText("ready")).toBeTruthy());
    expect(calls.every((c) => c.path === "/ready")).toBe(true);
    expect((screen.getByRole("button", { name: "Use token" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("shows the LLM as optional and unavailable without calling the system unhealthy", async () => {
    mockApi({ "GET /ready": { body: READY } });
    render(<App />);
    await waitFor(() => expect(screen.getByText(/LLM \(optional\)/)).toBeTruthy());
    expect(screen.getByText("ready")).toBeTruthy();
    expect(screen.getByText("unavailable")).toBeTruthy();
  });

  it("loads the overview after a token is entered and sends it as a bearer token", async () => {
    const { calls } = mockApi({ "GET /ready": { body: READY }, "GET /overview": { body: OVERVIEW } });
    render(<App />);
    await userEvent.type(screen.getByLabelText("API token"), "viewer-token");
    await userEvent.click(screen.getByRole("button", { name: "Use token" }));
    await waitFor(() => expect(screen.getByText("Awaiting approval")).toBeTruthy());
    expect(screen.getByText(/1 of 14 drifted: capital_loss/)).toBeTruthy();
    expect(screen.getByText(/Current performance is not measurable/)).toBeTruthy();
    expect(calls.find((c) => c.path === "/overview")!.headers.Authorization).toBe("Bearer viewer-token");
    expect(sessionStorage.getItem("aegis.token")).toBe("viewer-token");
  });

  it("shows an authentication error when the server rejects the token", async () => {
    mockApi({ "GET /ready": { body: READY },
      "GET /overview": { status: 401, body: { error: { code: "unauthorized", message: "invalid token" } } } });
    sessionStorage.setItem("aegis.token", "bad");
    render(<App />);
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("Error 401"));
    expect(screen.getByRole("alert").textContent).toContain("Enter a valid API token");
  });

  it("shows 'API unavailable' when nothing answers, without crashing", async () => {
    mockApi({});
    sessionStorage.setItem("aegis.token", "t");
    render(<App />);
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("API unavailable"));
    expect(screen.getByText("AEGIS")).toBeTruthy();
  });

  it("signing out clears the token and the data", async () => {
    mockApi({ "GET /ready": { body: READY }, "GET /overview": { body: OVERVIEW } });
    sessionStorage.setItem("aegis.token", "t");
    render(<App />);
    await waitFor(() => expect(screen.getByText("Awaiting approval")).toBeTruthy());
    await userEvent.click(screen.getByRole("button", { name: "Sign out" }));
    expect(sessionStorage.getItem("aegis.token")).toBeNull();
    expect(screen.queryByText("Awaiting approval")).toBeNull();
    expect(screen.getByText(/Enter an API token/)).toBeTruthy();
  });
});

describe("incident list", () => {
  beforeEach(() => { sessionStorage.setItem("aegis.token", "t"); window.location.hash = "#/incidents"; });

  it("renders incidents and opens one on click", async () => {
    mockApi({ "GET /ready": { body: READY }, "GET /incidents": { body: page([INCIDENT]) },
      "GET /incidents/INC-1": { body: INCIDENT } });
    render(<App />);
    await waitFor(() => expect(screen.getByText("INC-1")).toBeTruthy());
    // the status badge in the row (the same words also appear as a filter option)
    const badges = [...document.querySelectorAll("tbody .badge")].map((b) => b.textContent);
    expect(badges).toEqual(["◆medium", "◆awaiting approval"]);
    await userEvent.click(screen.getByText("data_quality"));
    await waitFor(() => expect(window.location.hash).toBe("#/incidents/INC-1"));
  });

  it("shows an explicit empty state", async () => {
    mockApi({ "GET /ready": { body: READY }, "GET /incidents": { body: page([]) } });
    render(<App />);
    await waitFor(() => expect(screen.getByText("No incidents have been detected.")).toBeTruthy());
  });

  it("filters by status through the API and resets paging", async () => {
    const { calls } = mockApi({ "GET /ready": { body: READY }, "GET /incidents": { body: page([INCIDENT], 40) } });
    render(<App />);
    await waitFor(() => expect(screen.getByText("1–1 of 40")).toBeTruthy());
    await userEvent.click(screen.getByRole("button", { name: "Next" }));
    await waitFor(() => expect(calls.some((c) => c.path.includes("offset=15"))).toBe(true));
    await userEvent.selectOptions(screen.getByRole("combobox"), "open");
    await waitFor(() => expect(calls.some((c) => c.path.includes("offset=0") && c.path.includes("status=open"))).toBe(true));
  });
});

describe("navigation", () => {
  it("reaches the evaluation page from the nav bar", async () => {
    mockApi({ "GET /ready": { body: READY }, "GET /overview": { body: OVERVIEW },
      "GET /evaluation/runs": { body: { runs: ["main"] } }, "GET /evaluation/runs/main": { body: EVAL_RUN } });
    sessionStorage.setItem("aegis.token", "t");
    render(<App />);
    await userEvent.click(screen.getByRole("link", { name: "Evaluation" }));
    await waitFor(() => expect(screen.getByText("Benchmark results (AEGIS-IR)")).toBeTruthy());
  });
});
