from fastapi.testclient import TestClient

from rml.api import app

client = TestClient(app)


def test_endpoints():
    assert client.get("/health").json()["status"] == "ok"
    names = [q["name"] for q in client.get("/queries").json()["rows"]]
    assert "path_to_conversion" in names
    assert client.get("/queries/new_to_brand").json()["row_count"] > 0
    assert client.get("/queries/nope").status_code == 404
    for path in ("/attribution", "/incrementality", "/roas"):
        assert client.get(path).json()["rows"], path
    plan = client.post("/optimize", json={"max_change": 0.1}).json()
    assert plan["max_change"] == 0.1 and len(plan["rows"]) == 4
    assert client.post("/optimize", json={"max_change": 2}).status_code == 422
    r = client.post("/ask", json={"question": "Which channel has the highest marginal ROAS?"})
    assert r.json()["tools_used"] == ["channel_roas"]


def test_api_key_is_enforced_when_set(monkeypatch):
    monkeypatch.setenv("RML_API_TOKEN", "s3cret")
    assert client.get("/roas").status_code == 401
    assert client.get("/roas", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/roas", headers={"X-API-Key": "s3cret"}).status_code == 200
    odd = {"X-API-Key": "caf\u00e9".encode("latin-1")}  # non-ASCII: a clean 401, not a 500
    assert client.get("/roas", headers=odd).status_code == 401
