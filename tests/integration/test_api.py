"""HTTP surface: authentication, validation, error shape, pagination, health/readiness."""

import io

import pytest
from sqlalchemy.exc import OperationalError

from aegis.observability.tracing import recent_spans
from tests.conftest import auth, open_incident

pytestmark = pytest.mark.postgres


def csv_bytes(frame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def upload(api, frame, role="current", dataset="adult", who="operator", filename="batch.csv"):
    return api.post(
        f"/data/{role}",
        data={"dataset": dataset},
        files={"file": (filename, csv_bytes(frame), "text/csv")},
        headers=auth(who),
    )


def test_health_is_public_and_ready_separates_required_from_optional(api):
    assert api.get("/health").json()["status"] == "ok"
    ready = api.get("/ready")
    body = ready.json()
    assert ready.status_code == 200 and body["status"] == "ready"
    assert body["components"]["database"] == {"status": "ok", "required": True, "detail": ""}
    assert (
        body["components"]["models"]["status"] == "ok" and body["components"]["models"]["required"]
    )
    assert (
        body["components"]["llm"]["status"] == "disabled"
        and body["components"]["llm"]["required"] is False
    )


def test_ready_reports_database_outage_while_health_stays_up(api, service, monkeypatch):
    monkeypatch.setattr(service.db, "ping", lambda: False)
    ready = api.get("/ready")
    assert ready.status_code == 503 and ready.json()["status"] == "not_ready"
    assert ready.json()["components"]["database"]["status"] == "unavailable"
    assert api.get("/health").status_code == 200


def test_ready_shows_llm_unavailable_without_failing_readiness(api, service):
    from tests.helpers import FakeOllama

    service.llm = FakeOllama(reachable=False).client()
    ready = api.get("/ready")
    assert ready.status_code == 200 and ready.json()["status"] == "ready"
    llm = ready.json()["components"]["llm"]
    assert (
        llm["status"] == "unavailable"
        and "deterministic diagnosis remains available" in llm["detail"]
    )


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/incidents"),
        ("get", "/overview"),
        ("get", "/audit"),
        ("get", "/models"),
        ("get", "/incidents/INC-x"),
        ("get", "/investigations/INV-x"),
        ("get", "/remediation/REM-x"),
        ("post", "/monitor/run"),
        ("post", "/incidents/INC-x/investigate"),
        ("post", "/incidents/INC-x/remediation"),
        ("post", "/remediation/REM-x/validate"),
        ("post", "/remediation/REM-x/approve"),
        ("post", "/remediation/REM-x/reject"),
        ("post", "/remediation/REM-x/rollback"),
        ("post", "/data/current"),
        ("post", "/data/reference"),
    ],
)
def test_every_data_endpoint_requires_a_valid_token(api, method, path):
    for headers in (
        {},
        {"Authorization": "Bearer wrong-token-000000000"},
        {"Authorization": "Basic abc"},
    ):
        response = getattr(api, method)(path, headers=headers)
        assert response.status_code == 401, (path, headers)
        assert response.json()["error"]["code"] == "unauthenticated"


def test_viewer_cannot_write_and_operator_cannot_decide(api):
    assert (
        api.post("/monitor/run", json={"dataset": "adult"}, headers=auth("viewer")).status_code
        == 403
    )
    assert api.get("/incidents", headers=auth("viewer")).status_code == 200
    denied = api.post(
        "/remediation/REM-x/approve", json={"expected_version": 1}, headers=auth("operator")
    )
    assert denied.status_code == 403 and denied.json()["error"]["code"] == "forbidden"


def test_errors_share_one_shape_and_carry_a_request_id(api):
    cases = [
        api.get("/incidents/INC-missing", headers=auth()),
        api.get("/nope", headers=auth()),
        api.post("/monitor/run", json={"dataset": "adult", "surprise": 1}, headers=auth()),
        api.post("/monitor/run", json={"dataset": "mnist"}, headers=auth()),
        api.get("/incidents?limit=100000", headers=auth()),
        api.post(
            "/monitor/run",
            content=b"{not json",
            headers={**auth(), "content-type": "application/json"},
        ),
    ]
    assert [c.status_code for c in cases] == [404, 404, 422, 422, 422, 422]
    for response in cases:
        error = response.json()["error"]
        assert set(error) == {"code", "message", "details", "request_id"} and error["request_id"]
        assert response.headers["x-request-id"] == error["request_id"]
    echoed = api.get("/incidents/INC-missing", headers={**auth(), "x-request-id": "trace-me-123"})
    assert echoed.json()["error"]["request_id"] == "trace-me-123"


def test_unexpected_errors_do_not_leak_internals(api, service, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret connection string postgres://user:hunter2@db")

    monkeypatch.setattr(service, "run_monitoring", boom)
    response = api.post("/monitor/run", json={"dataset": "adult"}, headers=auth())
    assert response.status_code == 500 and "hunter2" not in response.text
    assert response.json()["error"]["code"] == "internal_error"


def test_database_outage_returns_503_and_changes_nothing(api, service, monkeypatch):
    def down(*args, **kwargs):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    monkeypatch.setattr(service, "run_monitoring", down)
    response = api.post("/monitor/run", json={"dataset": "adult"}, headers=auth())
    assert (
        response.status_code == 503 and response.json()["error"]["code"] == "database_unavailable"
    )


def test_upload_validation_limits_and_safe_file_handling(api, ctx, service):
    ok = upload(api, ctx.clean(n_rows=300).current, filename="../../etc/passwd.csv")
    assert ok.status_code == 201 and ok.json()["rows"] == 300 and ok.json()["created_by"] == "bob"
    assert (
        ok.json()["source"] == "upload:....etcpasswd.csv"
    )  # name kept as text only, never used as a path
    assert upload(api, ctx.clean(n_rows=50).current, filename="data.parquet").status_code == 422
    assert upload(api, ctx.clean(n_rows=50).current, dataset="mnist").status_code == 422
    garbage = api.post(
        "/data/current",
        data={"dataset": "adult"},
        files={"file": ("x.csv", b"\x00\xff\x00\xff" * 100, "text/csv")},
        headers=auth(),
    )
    assert garbage.status_code == 422
    service.settings.max_upload_bytes = 2000
    too_big = upload(api, ctx.clean(n_rows=300).current)
    assert too_big.status_code == 422 and "exceeds" in too_big.json()["error"]["message"]
    service.settings.max_upload_bytes = 20 * 1024 * 1024
    service.settings.max_upload_rows = 100
    assert upload(api, ctx.clean(n_rows=300).current).status_code == 422
    service.settings.max_upload_rows = 200_000
    missing_target = upload(api, ctx.reference.drop(columns=["target"]).head(200), role="reference")
    assert missing_target.status_code == 422 and "target" in str(
        missing_target.json()["error"]["details"]
    )


def test_incident_listing_paginates_and_filters(api, service, ctx, principals):
    for seed, fault in enumerate(["F04", "F06", "F09"]):
        open_incident(service, ctx, principals, fault=fault, seed=seed + 20, investigate=False)
    page = api.get("/incidents?limit=2", headers=auth("viewer")).json()
    assert (
        page["total"] == 3
        and len(page["items"]) == 2
        and page["limit"] == 2
        and page["offset"] == 0
    )
    rest = api.get("/incidents?limit=2&offset=2", headers=auth("viewer")).json()
    assert len(rest["items"]) == 1 and {i["id"] for i in rest["items"]}.isdisjoint(
        {i["id"] for i in page["items"]}
    )
    detected = [i["detected_at"] for i in page["items"]]
    assert detected == sorted(detected, reverse=True)
    assert api.get("/incidents?status=mitigated", headers=auth("viewer")).json()["total"] == 0
    assert (
        api.get("/incidents?dataset=adult&status=open", headers=auth("viewer")).json()["total"] == 3
    )
    for item in page["items"]:
        assert set(item) >= {
            "id",
            "incident_type",
            "severity",
            "status",
            "detected_at",
            "model_version",
        }


def test_incident_and_investigation_details_expose_the_whole_chain(api, service, ctx, principals):
    incident, investigation = open_incident(service, ctx, principals)
    detail = api.get(f"/incidents/{incident.id}", headers=auth("viewer")).json()
    assert (
        detail["findings"]
        and detail["evidence"]
        and detail["performance"]["reference"]["roc_auc"] > 0.8
    )
    assert (
        detail["investigations"][0]["id"] == investigation.id
        and detail["remediations"][0]["state"] == "PENDING_APPROVAL"
    )
    inv = api.get(f"/investigations/{investigation.id}", headers=auth("viewer")).json()
    assert inv["llm_status"] == "disabled" and inv["hypotheses"][0]["root_cause"] == "scaling_error"
    assert (
        inv["tool_traces"][0]["duration_ms"] >= 0
        and inv["runbook_refs"]
        and inv["explanation"]["summary"]
    )
    assert {h["rank"] for h in inv["hypotheses"]} == set(range(1, len(inv["hypotheses"]) + 1))
    rem = api.get(f"/remediation/{detail['remediations'][0]['id']}", headers=auth("viewer")).json()
    assert (
        rem["executed"] is False
        and rem["approval"] is None
        and set(rem["allowed_events"]) == {"approve", "reject"}
    )


def test_overview_reports_real_aggregates_and_marks_unmeasurable_as_null(
    api, service, ctx, principals
):
    empty = api.get("/overview", headers=auth("viewer")).json()
    assert empty["incidents"] == {
        "total": 0,
        "active": 0,
        "critical_active": 0,
        "by_status": {},
        "by_severity": {},
    }
    adult = next(d for d in empty["datasets"] if d["dataset"] == "adult")
    assert adult["last_run"] is None and adult["deployment"]["model_version"] == ctx.record.version
    open_incident(service, ctx, principals, fault="F09", severity="high", investigate=False)
    after = api.get("/overview", headers=auth("viewer")).json()
    adult = next(d for d in after["datasets"] if d["dataset"] == "adult")
    assert after["incidents"]["total"] == 1 and after["incidents"]["active"] == 1
    assert (
        adult["last_run"]["performance"]["current"] is None
    )  # no labels: null, not a fabricated number
    assert adult["last_run"]["performance"]["reference"]["f1"] > 0.5
    assert adult["last_run"]["drift"]["features_tested"] == 12


def test_audit_models_metrics_and_latency_endpoints(api, service, ctx, principals):
    incident, _ = open_incident(service, ctx, principals)
    audit = api.get(f"/audit?incident_id={incident.id}&limit=200", headers=auth("viewer")).json()
    assert (
        audit["total"] == len(audit["items"]) >= 8
        and audit["items"][0]["action"] == "monitoring.run"
    )
    models = api.get("/models", headers=auth("viewer")).json()
    assert {m["version"] for m in models["adult"]} >= {"adult-test-v1", "adult-test-v0"}
    assert all(len(m["artifact_sha256"]) == 64 for m in models["adult"])
    metrics = api.get("/metrics").text
    for name in (
        "aegis_monitoring_seconds_bucket",
        "aegis_tool_calls_total",
        "aegis_incidents_total",
        "aegis_remediation_transitions_total",
        "aegis_process_rss_bytes",
    ):
        assert name in metrics
    latency = api.get("/observability/latency", headers=auth("viewer")).json()
    by_span = {s["span"]: s for s in latency["spans"]}
    assert (
        by_span["service.run_monitoring"]["p95_ms"]
        >= by_span["service.run_monitoring"]["p50_ms"]
        > 0
    )
    assert any(name.startswith("tool:") for name in by_span) and latency["process"]["rss_mb"] > 50


def test_api_requests_and_investigations_are_traced(api, service, ctx, principals):
    recent_spans.clear()
    incident, investigation = open_incident(service, ctx, principals)
    api.get(f"/incidents/{incident.id}", headers=auth("viewer"))
    http = [
        s
        for s in recent_spans.snapshot()
        if s["attributes"].get("http.route") == "/incidents/{incident_id}"
    ]
    assert (
        http
        and http[-1]["attributes"].get(
            "http.status_code", http[-1]["attributes"].get("http.response.status_code")
        )
        == 200
    )
    inv = api.get(f"/investigations/{investigation.id}", headers=auth("viewer")).json()
    spans = api.get(
        f"/observability/traces?trace_id={inv['trace_id']}", headers=auth("viewer")
    ).json()["spans"]
    names = {s["name"] for s in spans}
    assert {
        "service.investigation",
        "investigation.run",
        "investigation.rank_hypotheses",
        "tool.execute",
    } <= names


def test_investigate_endpoint_runs_in_background_and_is_pollable(api, service, ctx, principals):
    incident, _ = open_incident(service, ctx, principals, investigate=False)
    started = api.post(
        f"/incidents/{incident.id}/investigate", json={"use_llm": False}, headers=auth()
    )
    assert started.status_code == 202 and started.json()["status"] == "running"
    service._worker.shutdown(wait=True)
    done = api.get(f"/investigations/{started.json()['id']}", headers=auth("viewer")).json()
    assert done["status"] == "completed" and done["hypotheses"]
    assert api.post("/incidents/INC-nope/investigate", json={}, headers=auth()).status_code == 404
    assert (
        api.post(
            f"/incidents/{incident.id}/investigate", json={"run_shell": "id"}, headers=auth()
        ).status_code
        == 422
    )
