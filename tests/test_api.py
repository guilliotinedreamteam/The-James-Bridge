"""API tests — 47 tests."""
import json, pytest
from neurobridge.api import create_app
from neurobridge.config import Config
from tests.conftest import ecog as mk_ecog

@pytest.fixture
def c():
    app = create_app(); app.config["TESTING"] = True
    with app.test_client() as cl: yield cl

# ── Health ──
class TestHealth:
    def test_200(self, c): assert c.get("/api/health").status_code == 200
    def test_json(self, c): assert c.get("/api/health").get_json() is not None
    def test_ts(self, c): assert "timestamp" in c.get("/api/health").get_json()
    def test_ver(self, c): assert "version" in c.get("/api/health").get_json()
    def test_model_false(self, c): assert c.get("/api/health").get_json()["model_loaded"] is False
    def test_idempotent(self, c):
        r1, r2 = c.get("/api/health").get_json(), c.get("/api/health").get_json()
        assert r1["status"] == r2["status"] and r1["version"] == r2["version"]
    def test_post_405(self, c): assert c.post("/api/health").status_code == 405
    def test_put_405(self, c): assert c.put("/api/health").status_code == 405
    def test_delete_405(self, c): assert c.delete("/api/health").status_code == 405

# ── Process ──
def _post(c, data): return c.post("/api/process", data=json.dumps(data), content_type="application/json")

class TestProcess:
    def test_no_json(self, c): assert c.post("/api/process", content_type="application/json", data="").status_code == 400
    def test_no_signal(self, c): assert _post(c, {}).status_code == 400
    def test_null_signal(self, c): assert _post(c, {"signal": None}).status_code == 400
    def test_str_signal(self, c): assert _post(c, {"signal": "bad"}).status_code == 400
    def test_dict_signal(self, c): assert _post(c, {"signal": {"a": 1}}).status_code == 400
    def test_empty_signal(self, c): assert _post(c, {"signal": []}).status_code == 400
    def test_too_few(self, c):
        r = _post(c, {"signal": [0.1]*10})
        assert r.status_code == 400 and "SIGNAL_LENGTH_MISMATCH" in r.get_json().get("code", "")
    def test_too_many(self, c): assert _post(c, {"signal": [0.1]*(Config.NUM_FEATURES+50)}).status_code == 400
    def test_no_model_503(self, c):
        r = _post(c, {"signal": [0.1]*Config.NUM_FEATURES})
        assert r.status_code == 503 and r.get_json()["code"] == "MODEL_NOT_LOADED"
    def test_error_has_code(self, c):
        d = _post(c, {"signal": "bad"}).get_json()
        assert "code" in d and "error" in d
    def test_extra_fields_ok(self, c):
        assert _post(c, {"signal": [0.1]*Config.NUM_FEATURES, "extra": 42}).status_code in [200, 503]
    def test_get_405(self, c): assert c.get("/api/process").status_code == 405
    def test_wrong_ct(self, c):
        assert c.post("/api/process", data="x", content_type="application/x-www-form-urlencoded").status_code == 400
    def test_malformed_json(self, c):
        assert c.post("/api/process", data="{bad", content_type="application/json").status_code == 400

# ── Analyze ──
def _analyze(c, data, **kw): return c.post("/api/analyze", data=json.dumps({"data": data, **kw}), content_type="application/json")

class TestAnalyze:
    def test_no_json(self, c): assert c.post("/api/analyze", content_type="application/json", data="{}").status_code == 400
    def test_no_data(self, c): assert _analyze(c, None).status_code == 400
    def test_get_405(self, c): assert c.get("/api/analyze").status_code == 405
    def test_time_domain(self, c):
        d = _analyze(c, [[.1,.2],[.3,.4]], analysis_type="time_domain").get_json()
        assert d["results"]["analysis_type"] == "time_domain"
    def test_freq_domain(self, c):
        assert "frequency_analysis" in _analyze(c, [[.1,.2],[.3,.4]], analysis_type="frequency_domain").get_json()["results"]
    def test_stats(self, c):
        s = _analyze(c, [[1,2],[3,4]]).get_json()["results"]["statistics"]
        assert all(k in s for k in ["mean","std","min","max"])
    def test_stats_correct(self, c):
        s = _analyze(c, [[1,2,3,4]]).get_json()["results"]["statistics"]
        assert s["min"] == 1 and s["max"] == 4 and abs(s["mean"]-2.5) < .01
    def test_shape(self, c):
        assert _analyze(c, [[1,2],[3,4],[5,6]]).get_json()["results"]["shape"] == [3, 2]
    def test_single_val(self, c):
        assert _analyze(c, [42.0]).get_json()["results"]["statistics"]["mean"] == 42.0
    def test_default_type(self, c):
        assert _analyze(c, [1.0]).get_json()["results"]["analysis_type"] == "time_domain"
    def test_viz_data(self, c):
        assert _analyze(c, [1.0]).get_json()["visualization_data"]["available"] is True
    def test_realistic_ecog(self, c):
        e = mk_ecog(8, 10, seed=42)
        s = _analyze(c, e.tolist(), analysis_type="time_domain").get_json()["results"]["statistics"]
        assert 0 <= s["mean"] <= 1
    def test_freq_ecog(self, c):
        e = mk_ecog(8, 100, seed=42)
        assert _analyze(c, e.tolist(), analysis_type="frequency_domain").get_json()["results"]["frequency_analysis"]["total_power"] > 0

# ── Errors ──
class TestErrors:
    def test_404_json(self, c):
        d = c.get("/api/nonexistent").get_json()
        assert "error" in d and "code" in d
    def test_deep_404(self, c): assert c.get("/api/a/b/c").status_code == 404
    def test_root_404(self, c): assert c.get("/").status_code == 404
    def test_api_root_404(self, c): assert c.get("/api").status_code == 404
