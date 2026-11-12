// Real-browser verification of the dashboard against a running stack.
//
//   DASHBOARD_URL=http://localhost:8080 VIEWER_TOKEN=.. OPERATOR_TOKEN=.. APPROVER_TOKEN=.. \
//   INCIDENT_ID=INC-.. node e2e/browser-check.mjs
//
// Drives headless Chromium through the pages a reviewer uses, performs real interactions
// (sign-in, filtering, tab changes, a refused approval, a granted approval, a rollback), checks
// loading / empty / error states, repeats layout checks at a phone viewport, and writes a JSON
// report plus screenshots. Exit code 0 only if every check passed.
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { chromium } from "playwright";

const env = (name, fallback) => process.env[name] ?? fallback;
const BASE = env("DASHBOARD_URL", "http://localhost:8080");
const OUT = resolve(env("BROWSER_REPORT_DIR", "../../results/release/browser"));
const TOKENS = { viewer: env("VIEWER_TOKEN"), operator: env("OPERATOR_TOKEN"), approver: env("APPROVER_TOKEN") };
const INCIDENT = env("INCIDENT_ID");
const EVAL_RUN = env("EVAL_RUN", "main");
const results = {};
for (const [key, value] of Object.entries({ ...TOKENS, INCIDENT })) {
  if (!value) { console.error(`missing environment value for ${key}`); process.exit(2); }
}
mkdirSync(OUT, { recursive: true });

const checks = [];
const consoleErrors = [];
const pageErrors = [];
let page;

async function check(name, fn) {
  const started = Date.now();
  try {
    await fn();
    checks.push({ name, passed: true, ms: Date.now() - started });
    console.log(`  ok    ${name}`);
  } catch (error) {
    checks.push({ name, passed: false, ms: Date.now() - started, error: String(error?.message ?? error).slice(0, 600) });
    console.log(`  FAIL  ${name}\n        ${String(error?.message ?? error).split("\n")[0]}`);
    await page.screenshot({ path: `${OUT}/failure-${checks.length}.png` }).catch(() => {});
  }
}
const expect = (condition, message) => { if (!condition) throw new Error(message); };
const text = (selector) => page.locator(selector).first().innerText();
const visible = async (selector, timeout = 15000) => page.locator(selector).first().waitFor({ state: "visible", timeout });
const shot = (name) => page.screenshot({ path: `${OUT}/${name}.png`, fullPage: true });
async function signIn(role) {
  if (await page.getByRole("button", { name: "Sign out" }).count()) await page.getByRole("button", { name: "Sign out" }).click();
  await page.getByLabel("API token").fill(TOKENS[role]);
  await page.getByRole("button", { name: "Use token" }).click();
}
const noHorizontalOverflow = async () => {
  const [scroll, inner] = await page.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
  expect(scroll <= inner + 1, `page is ${scroll}px wide in a ${inner}px viewport (horizontal overflow)`);
};

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
page = await context.newPage();
page.on("console", (m) => { if (m.type() === "error") consoleErrors.push({ text: m.text(), url: page.url() }); });
page.on("pageerror", (e) => pageErrors.push({ text: String(e), url: page.url() }));

console.log(`browser ${browser.version()} -> ${BASE}`);

// ------------------------------------------------------------------ shell
await check("dashboard loads and asks for a token", async () => {
  const response = await page.goto(BASE, { waitUntil: "networkidle" });
  expect(response.status() === 200, `GET / -> ${response.status()}`);
  expect((await page.title()).includes("AEGIS"), "page title does not mention AEGIS");
  await visible("text=Enter an API token to load data");
  await visible(".health .badge");
});
await check("an invalid token produces a visible 401 error, not a blank page", async () => {
  await page.getByLabel("API token").fill("definitely-not-a-token");
  await page.getByRole("button", { name: "Use token" }).click();
  await visible("[role=alert]");
  const alert = await text("[role=alert]");
  expect(alert.includes("401"), `alert was: ${alert}`);
  expect(alert.includes("Enter a valid API token"), "401 hint missing");
  await shot("desktop-401");
});

// --------------------------------------------------------------- overview
await check("overview renders live data after sign-in", async () => {
  await signIn("viewer");
  await visible(".tiles .tile");
  const tiles = await page.locator(".tile").count();
  expect(tiles === 5, `expected 5 tiles, saw ${tiles}`);
  const total = Number(await text(".tile >> nth=0 >> .tile-value"));
  expect(Number.isInteger(total) && total >= 1, `total incidents tile shows '${total}'`);
  await visible("text=System health");
  expect((await text("text=Dataset: adult")).length > 0, "dataset card missing");
  await shot("desktop-overview");
});
await check("loading state is shown while the API is slow", async () => {
  await page.route("**/api/overview", async (route) => { await new Promise((r) => setTimeout(r, 1500)); await route.continue(); });
  await page.reload();
  await visible("[aria-busy=true]", 5000);
  await visible(".tiles .tile");
  await page.unroute("**/api/overview");
});
await check("an API failure shows 'API unavailable' and the shell stays usable", async () => {
  await page.route("**/api/overview", (route) => route.abort());
  await page.reload();
  await visible("[role=alert]");
  expect((await text("[role=alert]")).includes("API unavailable"), "no 'API unavailable' message");
  await visible("nav a >> text=Incidents");
  await page.unroute("**/api/overview");
});
await check("a server error (500) is reported with its status", async () => {
  await page.route("**/api/overview", (route) => route.fulfill({ status: 500, contentType: "application/json",
    body: JSON.stringify({ error: { code: "internal_error", message: "an unexpected error occurred", request_id: "req-browser" } }) }));
  await page.reload();
  await visible("[role=alert]");
  const alert = await text("[role=alert]");
  expect(alert.includes("Error 500") && alert.includes("req-browser"), `alert was: ${alert}`);
  await page.unroute("**/api/overview");
});

// -------------------------------------------------------------- incidents
await check("incident list renders and links to the incident", async () => {
  await page.click("nav a >> text=Incidents");
  await visible("table.clickable tbody tr");
  await visible(`a[href='#/incidents/${INCIDENT}']`);
  await shot("desktop-incidents");
});
await check("status filter with no matches shows the empty state", async () => {
  await page.locator("select").selectOption("investigating");
  await visible("text=No incidents with this status.");
  await page.locator("select").selectOption("");
  await visible("table.clickable tbody tr");
});
await check("incident detail: findings and evidence panels render", async () => {
  await page.click(`a[href='#/incidents/${INCIDENT}']`);
  await visible("text=Monitoring findings");
  expect(await page.locator("section.card >> nth=0 >> tbody tr").count() >= 1, "no finding rows");
  await visible("text=Feature-level findings");
  await visible("text=Performance change");
  const evidenceTitle = await text("h3:has-text('Evidence records')");
  expect(/Evidence records \([1-9]/.test(evidenceTitle), `evidence card title: ${evidenceTitle}`);
  await shot("desktop-incident-evidence");
});
await check("investigation tab: ranked hypotheses with evidence, worded as hypotheses", async () => {
  await page.getByRole("tab", { name: "Investigation" }).click();
  await visible(".hypothesis");
  await visible("text=hypotheses, not established causes");
  expect((await text(".hypothesis b")).length > 0, "hypothesis has no cause label");
  expect((await text(".explanation")).includes("deterministic"), "explanation source not shown");
  await page.locator(".hypothesis details summary").first().click();
  await visible(".hypothesis details[open] li code");
  await visible("text=Tool calls");
  await shot("desktop-incident-investigation");
});
await check("remediation tab: proposal shown as recommendation with validation and policy", async () => {
  await page.getByRole("tab", { name: "Remediation" }).click();
  await visible(".banner-proposal");
  expect((await text(".banner")).includes("RECOMMENDATION ONLY"), "pending proposal not labelled as recommendation");
  await visible("text=Sandbox validation");
  await visible("text=Policy decision");
  await visible("text=Waiting for a human approver.");
  const done = await page.locator("ol.lifecycle li.done").allInnerTexts();
  expect(done.join("|").includes("pending approval") && !done.join("|").includes("executed"), `lifecycle: ${done}`);
  await shot("desktop-incident-remediation-pending");
});
await check("a viewer's approval is refused by the server and shown as refused", async () => {
  await page.getByRole("button", { name: "Approve and execute" }).click();
  await visible("p.error[role=alert]");
  expect((await text("p.error[role=alert]")).includes("403"), "refusal status not shown");
  expect((await text(".banner")).includes("RECOMMENDATION ONLY"), "UI claims execution after a refused approval");
});
await check("an approver approves: UI shows executed, verified, and the audit trail", async () => {
  await signIn("approver");
  await visible("text=Monitoring findings");
  await page.getByRole("tab", { name: "Remediation" }).click();
  await page.getByPlaceholder("Decision comment").fill("browser verification");
  await page.getByRole("button", { name: "Approve and execute" }).click();
  await visible(".banner-executed", 60000);
  expect((await text(".banner")).includes("EXECUTED"), "no executed banner");
  await visible("text=Post-execution verification");
  expect((await text("text=approved by")).includes("browser verification"), "approval comment not shown");
  await shot("desktop-incident-remediation-executed");
  await page.getByRole("tab", { name: "Audit trail" }).click();
  await visible("code:text('remediation.execute')");
  await visible("td .badge:has-text('denied')");
  await shot("desktop-incident-audit");
});
await check("rollback needs a reason and then restores the previous state", async () => {
  await page.getByRole("tab", { name: "Remediation" }).click();
  const rollback = page.getByRole("button", { name: "Roll back" });
  expect(await rollback.isDisabled(), "rollback enabled without a reason");
  await page.getByPlaceholder("Rollback reason (required)").fill("browser verification rollback");
  await rollback.click();
  await visible("text=ROLLED BACK", 60000);
});
await check("a non-existent incident shows a not-found error", async () => {
  await page.goto(`${BASE}/#/incidents/INC-does-not-exist`);
  await visible("[role=alert]");
  expect((await text("[role=alert]")).includes("404"), "no 404 shown");
});

// ------------------------------------------------------------- evaluation
await check("evaluation page: both detection metrics under their own names and populations", async () => {
  await page.click("nav a >> text=Evaluation");
  await visible("select[aria-label='Benchmark run']");
  await page.locator("select[aria-label='Benchmark run']").selectOption(EVAL_RUN);
  await visible("[data-testid=overall_fault_detection_f1]");
  // What the page shows must be what the API serves for this run, each value under its own name.
  const api = await (await page.request.get(`${BASE}/api/evaluation/runs/${EVAL_RUN}`,
    { headers: { Authorization: `Bearer ${TOKENS.viewer}` } })).json();
  const [overallApi, driftApi] = api.detection;
  expect(overallApi.metric_prefix === "overall_fault_detection" && driftApi.metric_prefix === "drift_detection", "API metric order/prefix");
  const overall = await text("[data-testid=overall_fault_detection_f1]");
  const drift = await text("[data-testid=drift_detection_f1]");
  const overallF1 = overallApi.metrics.overall_fault_detection_f1.value.toFixed(4);
  const driftF1 = driftApi.metrics.drift_detection_f1.value.toFixed(4);
  expect(overall.startsWith(overallF1), `overall_fault_detection_f1 shows '${overall}', API says ${overallF1}`);
  expect(drift.startsWith(driftF1), `drift_detection_f1 shows '${drift}', API says ${driftF1}`);
  expect(await text("[data-testid=overall_fault_detection-positives]") === String(overallApi.population.n_positive_scenarios), "overall positives differ from API");
  expect(await text("[data-testid=drift_detection-positives]") === String(driftApi.population.n_positive_scenarios), "drift positives differ from API");
  expect(overallApi.population.n_positive_scenarios > driftApi.population.n_positive_scenarios, "populations should differ");
  await visible("text=their values are not interchangeable");
  await visible("[data-testid=reconciled]");
  expect((await text("[data-testid=reconciled]")).includes("reconciled"), "lifecycle reconciliation status missing");
  results.evaluation = { run: EVAL_RUN, overall_fault_detection_f1: overallF1, drift_detection_f1: driftF1 };
  await shot("desktop-evaluation");
});
await check("evaluation page: switching the run reloads the results", async () => {
  const options = await page.locator("select[aria-label='Benchmark run'] option").allInnerTexts();
  expect(options.length >= 1, "no benchmark runs offered");
  for (const option of options) {
    await page.locator("select[aria-label='Benchmark run']").selectOption(option);
    await visible("[data-testid=overall_fault_detection_f1]");
    await page.waitForFunction((o) => document.querySelector("p.muted.small code")?.textContent === o, option, { timeout: 15000 });
  }
});
await check("desktop layout has no horizontal overflow", noHorizontalOverflow);

// ------------------------------------------------------------ small screen
await page.setViewportSize({ width: 390, height: 844 });
for (const [name, go] of [
  ["overview", async () => { await page.goto(`${BASE}/#/`); await visible(".tiles .tile"); }],
  ["incidents", async () => { await page.goto(`${BASE}/#/incidents`); await visible("table.clickable tbody tr"); }],
  ["incident", async () => { await page.goto(`${BASE}/#/incidents/${INCIDENT}`); await visible("text=Monitoring findings"); }],
  ["evaluation", async () => { await page.goto(`${BASE}/#/evaluation`); await visible("[data-testid=drift_detection_f1]"); }],
]) {
  await check(`phone viewport (390px): ${name} renders without horizontal page overflow`, async () => {
    await go();
    await noHorizontalOverflow();
    await visible("nav a >> text=Evaluation");
    await shot(`phone-${name}`);
  });
}
await check("phone viewport: tabs remain operable", async () => {
  await page.goto(`${BASE}/#/incidents/${INCIDENT}`);
  await page.getByRole("tab", { name: "Remediation" }).click();
  await visible("text=ROLLED BACK");
  await noHorizontalOverflow();
});

// ----------------------------------------------------------- console audit
// Chromium logs every non-2xx response as a console error. The 4xx/5xx responses this script
// provokes on purpose are expected; anything else is a real defect.
const EXPECTED = /Failed to load resource: (the server responded with a status of (401|403|404|500)|net::ERR_FAILED)/;
const unexpected = consoleErrors.filter((e) => !EXPECTED.test(e.text));
await check("no unexpected console errors and no uncaught exceptions", async () => {
  expect(pageErrors.length === 0, `uncaught exceptions: ${JSON.stringify(pageErrors).slice(0, 400)}`);
  expect(unexpected.length === 0, `console errors: ${JSON.stringify(unexpected).slice(0, 400)}`);
});

const report = {
  generated_at: new Date().toISOString(),
  dashboard_url: BASE,
  browser: `chromium ${browser.version()} (headless)`,
  viewports: ["1440x900", "390x844"],
  incident_id: INCIDENT,
  observed: results,
  passed: checks.every((c) => c.passed),
  checks_total: checks.length,
  checks_failed: checks.filter((c) => !c.passed).length,
  checks,
  console_errors_expected: consoleErrors.length - unexpected.length,
  console_errors_unexpected: unexpected,
  uncaught_exceptions: pageErrors,
  limits: [
    "Headless Chromium only; Firefox and Safari were not exercised.",
    "Layout is checked for horizontal overflow and element visibility, not judged visually; screenshots are saved for a human to review.",
  ],
};
writeFileSync(`${OUT}/browser-report.json`, JSON.stringify(report, null, 2) + "\n");
await browser.close();
console.log(`\n${report.checks_total - report.checks_failed}/${report.checks_total} checks passed -> ${OUT}/browser-report.json`);
process.exit(report.passed ? 0 : 1);
