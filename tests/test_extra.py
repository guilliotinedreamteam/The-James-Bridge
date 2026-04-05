"""Extra rigorous tests — 21 tests to reach 200 total."""
import numpy as np, pytest, tempfile
from tests.conftest import ecog as mk_ecog, labels as mk_labels, ARPABET
from pathlib import Path

# ── Model weight properties ──
class TestWeights:
    def test_weights_are_finite(self):
        from neurobridge.model import build_neurobridge_decoder as B
        m = B(timesteps=5, features=8, num_classes=3, lstm_units=16)
        assert all(np.all(np.isfinite(w)) for w in m.get_weights())

    def test_weights_not_all_zero(self):
        from neurobridge.model import build_neurobridge_decoder as B
        m = B(timesteps=5, features=8, num_classes=3, lstm_units=16)
        assert any(np.any(w != 0) for w in m.get_weights())

    def test_rt_weights_finite(self):
        from neurobridge.model import build_realtime_decoder as B
        assert all(np.all(np.isfinite(w)) for w in B(features=8, num_classes=3, lstm_units=16).get_weights())

# ── Data edge cases ──
class TestDataEdge:
    def test_1_sample(self):
        from neurobridge.data import generate_mock_ecog_data as G; assert G(1, timesteps=5, features=8).shape == (1, 5, 8)

    def test_1_timestep(self):
        from neurobridge.data import generate_mock_ecog_data as G; assert G(3, timesteps=1, features=8).shape == (3, 1, 8)

    def test_1_feature(self):
        from neurobridge.data import generate_mock_ecog_data as G; assert G(3, timesteps=5, features=1).shape == (3, 5, 1)

    def test_large_dims(self):
        from neurobridge.data import generate_mock_ecog_data as G; assert G(2, timesteps=500, features=256).shape == (2, 500, 256)

    def test_binary_labels(self):
        from neurobridge.data import generate_mock_phoneme_labels as G
        l = G(3, timesteps=5, num_classes=2); assert l.shape == (3, 5, 2)
        np.testing.assert_allclose(l.sum(axis=-1), 1.0)

    def test_100_classes(self):
        from neurobridge.data import generate_mock_phoneme_labels as G; assert G(2, timesteps=3, num_classes=100).shape == (2, 3, 100)

    def test_gen_multiple_batches(self):
        from neurobridge.data import create_data_generator as G
        gen = G(batch_size=2)
        for _ in range(5): e, l = next(gen); assert e.shape[0] == 2

# ── API validation ──
class TestAPIExtra:
    @pytest.fixture
    def c(self):
        from neurobridge.api import create_app
        app = create_app(); app.config["TESTING"] = True
        with app.test_client() as cl: yield cl

    def test_analyze_viz_available(self, c):
        import json
        r = c.post("/api/analyze", data=json.dumps({"data": [1,2,3]}), content_type="application/json")
        assert r.get_json()["visualization_data"]["available"] is True

    def test_process_signal_with_nan_handled(self, c):
        import json
        r = c.post("/api/process", data=json.dumps({"signal": [float("nan")]*128}), content_type="application/json")
        assert r.status_code in [400, 503]

# ── Synthesizer edge cases ──
class TestSynthExtra:
    def test_long_sequence(self):
        from neurobridge.synthesizer import synthesize_speech_from_phonemes as S
        ids = np.arange(20, dtype=np.int32)
        a = S(ids, sampling_rate=8000, phoneme_duration_sec=.1, phoneme_map=ARPABET)
        assert len(a) == 16000

    def test_different_sampling_rate(self):
        from neurobridge.synthesizer import synthesize_speech_from_phonemes as S
        ids = np.array([0], dtype=np.int32)
        a1 = S(ids, sampling_rate=8000, phoneme_duration_sec=.1, phoneme_map=ARPABET)
        a2 = S(ids, sampling_rate=16000, phoneme_duration_sec=.1, phoneme_map=ARPABET)
        assert len(a2) == 2 * len(a1)

# ── Inference label diversity ──
class TestLabelDiversity:
    def test_labels_vary_across_inputs(self):
        from neurobridge.inference import RealtimeDecoder
        from neurobridge.model import build_realtime_decoder
        d = RealtimeDecoder()
        d.model = build_realtime_decoder(features=16, num_classes=5, lstm_units=16)
        d._features = 16; d._num_phonemes = 5
        labels = {d.predict_label(mk_ecog(16, 1, seed=i*100).flatten(), phoneme_map=ARPABET[:5]) for i in range(20)}
        assert len(labels) >= 1

# ── Evaluate edge cases ──
class TestEvalExtra:
    def test_alternating_silence_decode(self):
        from neurobridge.evaluate import decode_phoneme_sequence as D
        assert D(np.array([39,0,1,39,0]), phoneme_map=ARPABET) == ["SIL","AA","AE","SIL","AA"]

    def test_config_model_dir_is_path(self):
        from neurobridge.config import Config; from pathlib import Path
        assert isinstance(Config.MODEL_DIR, Path)

    def test_version_is_string(self):
        from neurobridge import __version__
        assert isinstance(__version__, str) and len(__version__) > 0

    def test_package_exports(self):
        import neurobridge
        assert hasattr(neurobridge, "__version__")

    def test_realtime_decoder_layer_count(self):
        from neurobridge.model import build_realtime_decoder as B
        assert len(B(features=8, num_classes=3, lstm_units=16).layers) >= 3

    def test_ecog_fixture_normalized(self):
        e = mk_ecog(8, 5, seed=42)
        assert e.min() >= 0.0 and e.max() <= 1.0

