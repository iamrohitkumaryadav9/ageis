import { useState } from "react";
import { api } from "./api";
import type { Ready } from "./types";
import { Card, MetricBars, Severity, StateBox, Status, Tile, fmtMs, fmtTime, useAsync } from "./ui";

const METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"];

export function OverviewPage({ ready }: { ready: Ready | null }) {
  const { data, error, loading } = useAsync(() => api.overview(), [], 10000);
  return (
    <StateBox loading={loading && !data} error={error}>
      {data && (
        <>
          <div className="tiles">
            <Tile label="Total incidents" value={data.incidents.total} />
            <Tile label="Active incidents" value={data.incidents.active} />
            <Tile label="Critical, active" value={data.incidents.critical_active} />
            <Tile label="Awaiting approval" value={data.pending_approvals} note="validated proposals, not executed" />
            <Tile label="Monitoring runs" value={data.monitoring_runs} />
          </div>
          <div className="grid-2">
            {data.datasets.map((d) => (
              <Card key={d.dataset} title={`Dataset: ${d.dataset}`}
                aside={d.last_run ? <Status value={d.last_run.incident_detected ? "open" : "ok"} /> : undefined}>
                {!d.deployment ? <p className="muted">No deployment registered. Run <code>make bootstrap</code>.</p> : (
                  <dl className="kv">
                    <dt>Model version</dt><dd><code>{d.deployment.model_version}</code></dd>
                    <dt>Deployment revision</dt><dd>#{d.deployment.revision}{d.deployment.quarantined ? " · batch quarantined" : ""}</dd>
                  </dl>
                )}
                {!d.last_run ? <p className="muted">Monitoring has not run for this dataset yet.</p> : (
                  <>
                    <dl className="kv">
                      <dt>Last monitoring run</dt><dd>{fmtTime(d.last_run.at)} · {d.last_run.rows} rows · {fmtMs(d.last_run.duration_ms)}</dd>
                      <dt>Drift status</dt>
                      <dd>{d.last_run.drift.features_drifted.length === 0
                        ? `no drift in ${d.last_run.drift.features_tested} features tested`
                        : `${d.last_run.drift.features_drifted.length} of ${d.last_run.drift.features_tested} drifted: ${d.last_run.drift.features_drifted.join(", ")}`}</dd>
                      <dt>Worst severity</dt><dd><Severity value={d.last_run.max_severity} /></dd>
                    </dl>
                    <h4>Model performance (labelled rows: {d.last_run.performance.labelled_rows})</h4>
                    {d.last_run.performance.reference ? (
                      <MetricBars rows={METRICS.map((m) => ({ name: m, reference: d.last_run!.performance.reference![m] ?? null,
                        current: d.last_run!.performance.current ? d.last_run!.performance.current[m] ?? null : null }))} />
                    ) : <p className="muted">Performance could not be computed for this run (the batch could not be scored).</p>}
                    {!d.last_run.performance.current && d.last_run.performance.reference &&
                      <p className="muted">Current performance is not measurable: too few labels have arrived. No value is shown rather than a guess.</p>}
                  </>
                )}
              </Card>
            ))}
          </div>
          <Card title="System health">
            {!ready ? <p className="muted">Health information unavailable.</p> : (
              <table><thead><tr><th>Component</th><th>Status</th><th>Required</th><th>Detail</th></tr></thead>
                <tbody>{Object.entries(ready.components).map(([name, c]) => (
                  <tr key={name}><td>{name}</td><td><Status value={c.status} /></td><td>{c.required ? "yes" : "optional"}</td><td>{c.detail || "—"}</td></tr>
                ))}</tbody></table>
            )}
          </Card>
        </>
      )}
    </StateBox>
  );
}

const PAGE = 15;
const STATUSES = ["", "open", "investigating", "investigated", "awaiting_approval", "mitigated"];

export function IncidentList() {
  const [offset, setOffset] = useState(0);
  const [status, setStatus] = useState("");
  const { data, error, loading } = useAsync(() => api.incidents(PAGE, offset, status), [offset, status], 10000);
  return (
    <Card title="Incidents" aside={
      <label className="filter">Status
        <select value={status} onChange={(e) => { setStatus(e.target.value); setOffset(0); }}>
          {STATUSES.map((s) => <option key={s} value={s}>{s ? s.replace(/_/g, " ") : "all"}</option>)}
        </select>
      </label>}>
      <StateBox loading={loading && !data} error={error} empty={data?.items.length === 0}
        emptyText={status ? "No incidents with this status." : "No incidents have been detected."}>
        {data && (
          <>
            <table className="clickable">
              <thead><tr><th>Incident</th><th>Type</th><th>Severity</th><th>Status</th><th>Detected</th><th>Dataset</th><th>Model version</th></tr></thead>
              <tbody>{data.items.map((i) => (
                <tr key={i.id} onClick={() => { window.location.hash = `#/incidents/${i.id}`; }}>
                  <td><a href={`#/incidents/${i.id}`}><code>{i.id}</code></a></td><td>{i.incident_type}</td>
                  <td><Severity value={i.severity} /></td><td><Status value={i.status} /></td>
                  <td>{fmtTime(i.detected_at)}</td><td>{i.dataset}</td><td><code>{i.model_version}</code></td>
                </tr>))}</tbody>
            </table>
            <div className="pager">
              <button className="ghost" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>Previous</button>
              <span>{offset + 1}–{offset + data.items.length} of {data.total}</span>
              <button className="ghost" disabled={offset + PAGE >= data.total} onClick={() => setOffset(offset + PAGE)}>Next</button>
            </div>
          </>
        )}
      </StateBox>
    </Card>
  );
}
