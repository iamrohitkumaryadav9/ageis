import { useEffect, useState } from "react";
import { api, getToken, setToken } from "./api";
import { IncidentView } from "./IncidentView";
import { IncidentList, OverviewPage } from "./Pages";
import { Status, useAsync } from "./ui";

type Route = { page: "overview" } | { page: "incidents" } | { page: "incident"; id: string };

function parseRoute(): Route {
  const [, page, id] = window.location.hash.split("/");
  if (page === "incidents" && id) return { page: "incident", id };
  if (page === "incidents") return { page: "incidents" };
  return { page: "overview" };
}

export function App() {
  const [route, setRoute] = useState<Route>(parseRoute);
  const [token, setTokenState] = useState(getToken);
  const [draft, setDraft] = useState("");
  useEffect(() => {
    const onHash = () => setRoute(parseRoute());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const ready = useAsync(() => api.ready(), [], 15000);

  const saveToken = (value: string) => { setToken(value); setTokenState(value); setDraft(""); };
  const llm = ready.data?.components.llm;

  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="#/"><b>AEGIS</b><span>ML Incident Commander</span></a>
        <nav>
          <a href="#/" className={route.page === "overview" ? "active" : ""}>Overview</a>
          <a href="#/incidents" className={route.page !== "overview" ? "active" : ""}>Incidents</a>
        </nav>
        <div className="health" title={Object.entries(ready.data?.components ?? {}).map(([k, v]) => `${k}: ${v.status} ${v.detail}`).join("\n")}>
          {ready.error ? <Status value="unavailable" /> : ready.data ? (
            <>
              <span className="health-item">system <Status value={ready.data.status} /></span>
              <span className="health-item">LLM (optional) <Status value={llm?.status} /></span>
            </>
          ) : <span className="muted">checking…</span>}
        </div>
        <form className="token" onSubmit={(e) => { e.preventDefault(); saveToken(draft.trim()); }}>
          {token ? (
            <button type="button" className="ghost" onClick={() => saveToken("")}>Sign out</button>
          ) : (
            <>
              <input type="password" placeholder="API token" value={draft} onChange={(e) => setDraft(e.target.value)} aria-label="API token" autoComplete="off" />
              <button type="submit" disabled={!draft.trim()}>Use token</button>
            </>
          )}
        </form>
      </header>
      <main key={token /* reload every view when the identity changes */}>
        {!token && <div className="state">Enter an API token to load data. Tokens are configured on the server (AEGIS_API_TOKENS).</div>}
        {token && route.page === "overview" && <OverviewPage ready={ready.data} />}
        {token && route.page === "incidents" && <IncidentList />}
        {token && route.page === "incident" && <IncidentView id={route.id} />}
      </main>
    </div>
  );
}
