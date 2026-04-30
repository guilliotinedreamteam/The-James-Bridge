"""Core module tests — config, data, evaluate, inference, synthesizer, integration. 115 tests."""
import numpy as np, pytest, tempfile, os
from unittest.mock import patch
from tests.conftest import ecog as mk_ecog, labels as mk_labels, ARPABET

# ═══ CONFIG (39 tests) ═══
class TestConfigDefaults:
    def test_timesteps(self):
        from neurobridge.config import Config as C; assert C.NUM_TIMESTEPS == 100
    def test_features(self):
        from neurobridge.config import Config as C; assert C.NUM_FEATURES == 128
    def test_phonemes(self):
        from neurobridge.config import Config as C; assert C.NUM_PHONEMES == 41
    def test_lstm(self):
        from neurobridge.config import Config as C; assert C.LSTM_UNITS == 256
    def test_dense(self):
        from neurobridge.config import Config as C; assert C.DENSE_UNITS == 128
    def test_batch(self):
        from neurobridge.config import Config as C; assert C.BATCH_SIZE == 32
    def test_epochs(self):
        from neurobridge.config import Config as C; assert C.EPOCHS == 5
    def test_lr(self):
        from neurobridge.config import Config as C; assert C.LEARNING_RATE == 0.001
    def test_port(self):
        from neurobridge.config import Config as C; assert C.API_PORT == 5000
    def test_host(self):
        from neurobridge.config import Config as C; assert C.API_HOST == "0.0.0.0"
    def test_sr(self):
        from neurobridge.config import Config as C; assert C.AUDIO_SAMPLING_RATE == 16000
    def test_dur(self):
        from neurobridge.config import Config as C; assert C.PHONEME_DURATION_SEC == 0.1

class TestConfigPaths:
    def test_model_ext(self):
        from neurobridge.config import Config as C; assert str(C.model_save_path()).endswith(".keras")
    def test_rt_ext(self):
        from neurobridge.config import Config as C; assert str(C.realtime_model_save_path()).endswith(".keras")
    def test_dir_created(self, tmp_path):
        from neurobridge.config import Config as C
        orig = C.MODEL_DIR; C.MODEL_DIR = tmp_path / "m"
        try: C.model_save_path(); assert C.MODEL_DIR.exists()
        finally: C.MODEL_DIR = orig

class TestConfigSummary:
    def test_str(self):
        from neurobridge.config import Config as C; assert isinstance(C.summary(), str)
    def test_fields(self):
        from neurobridge.config import Config as C
        s = C.summary(); assert all(k in s for k in ["Timesteps","Features","Phonemes","LSTM"])

class TestPhonemeMap:
    def test_len(self):
        from neurobridge.config import Config as C; assert len(C.PHONEME_MAP) == C.NUM_PHONEMES
    def test_prefix(self):
        from neurobridge.config import Config as C
        assert all(l.startswith("PH_") for l in C.PHONEME_MAP)

class TestEnvOverrides:
    def _reload(self):
        import importlib, neurobridge.config as cfg; importlib.reload(cfg); return cfg.Config

    def test_ts(self):
        with patch.dict(os.environ, {"NB_NUM_TIMESTEPS": "200"}):
            try: assert self._reload().NUM_TIMESTEPS == 200
            finally: self._reload()
    def test_feat(self):
        with patch.dict(os.environ, {"NB_NUM_FEATURES": "64"}):
            try: assert self._reload().NUM_FEATURES == 64
            finally: self._reload()
    def test_port(self):
        with patch.dict(os.environ, {"NB_API_PORT": "8080"}):
            try: assert self._reload().API_PORT == 8080
            finally: self._reload()
    def test_lr(self):
        with patch.dict(os.environ, {"NB_LEARNING_RATE": "0.01"}):
            try: assert self._reload().LEARNING_RATE == 0.01
            finally: self._reload()

class TestConfigConsistency:
    def test_map_matches(self):
        from neurobridge.config import Config as C; assert len(C.PHONEME_MAP) == C.NUM_PHONEMES
    def test_lr_pos(self):
        from neurobridge.config import Config as C; assert C.LEARNING_RATE > 0
    def test_batch_pos(self):
        from neurobridge.config import Config as C; assert C.BATCH_SIZE > 0
    def test_epochs_pos(self):
        from neurobridge.config import Config as C; assert C.EPOCHS > 0
    def test_ts_pos(self):
        from neurobridge.config import Config as C; assert C.NUM_TIMESTEPS > 0
    def test_feat_pos(self):
        from neurobridge.config import Config as C; assert C.NUM_FEATURES > 0
    def test_ph_pos(self):
        from neurobridge.config import Config as C; assert C.NUM_PHONEMES > 0
    def test_lstm_pos(self):
        from neurobridge.config import Config as C; assert C.LSTM_UNITS > 0
    def test_host_str(self):
        from neurobridge.config import Config as C; assert isinstance(C.API_HOST, str)
    def test_sr_range(self):
        from neurobridge.config import Config as C; assert 8000 <= C.AUDIO_SAMPLING_RATE <= 48000
    def test_dur_pos(self):
        from neurobridge.config import Config as C; assert 0 < C.PHONEME_DURATION_SEC < 1.0
    def test_model_dir_type(self):
        from neurobridge.config import Config as C; from pathlib import Path
        assert isinstance(C.MODEL_DIR, (str, Path))

# ═══ DATA (12 tests) ═══
class TestData:
    def test_shape(self):
        from neurobridge.data import generate_mock_ecog_data as G; assert G(5, timesteps=10, features=16).shape == (5, 10, 16)
    def test_dtype(self):
        from neurobridge.data import generate_mock_ecog_data as G; assert G(3, timesteps=5, features=8).dtype == np.float32
    def test_bounds(self):
        from neurobridge.data import generate_mock_ecog_data as G; d = G(10, timesteps=5, features=8); assert d.min() >= 0 and d.max() <= 1
    def test_finite(self):
        from neurobridge.data import generate_mock_ecog_data as G; assert np.all(np.isfinite(G(10, timesteps=50, features=32)))
    def test_varies(self):
        from neurobridge.data import generate_mock_ecog_data as G; d = G(5, timesteps=10, features=8); assert not np.array_equal(d[0], d[1])
    def test_lbl_shape(self):
        from neurobridge.data import generate_mock_phoneme_labels as G; assert G(5, timesteps=10, num_classes=3).shape == (5, 10, 3)
    def test_lbl_onehot(self):
        from neurobridge.data import generate_mock_phoneme_labels as G; np.testing.assert_allclose(G(5, timesteps=10, num_classes=3).sum(axis=-1), 1.0)
    def test_lbl_vals(self):
        from neurobridge.data import generate_mock_phoneme_labels as G; assert set(np.unique(G(10, timesteps=20, num_classes=5))).issubset({0.0, 1.0})
    def test_gen_batch(self):
        from neurobridge.data import create_data_generator as G
        e, l = next(G(batch_size=4)); assert e.shape[0] == 4 and l.shape[0] == 4
    def test_gen_match(self):
        from neurobridge.data import create_data_generator as G
        e, l = next(G(batch_size=4)); assert e.shape[0] == l.shape[0] and e.shape[1] == l.shape[1]
    def test_gen_dtype(self):
        from neurobridge.data import create_data_generator as G
        e, l = next(G(batch_size=3)); assert e.dtype == np.float32 and l.dtype == np.float32
    def test_real_loader_missing(self):
        from neurobridge.data import load_real_ecog_data as L; assert L("/nonexistent/x.mat") is None

# ═══ EVALUATE (18 tests) ═══
class TestEvaluate:
    def _m(self):
        from neurobridge.model import build_neurobridge_decoder as B, compile_model as C
        m = B(timesteps=5, features=8, num_classes=3, lstm_units=16); C(m); return m

    def test_accuracy(self, ecog_4x5x8, labels_4x5x3):
        from neurobridge.evaluate import evaluate_model as E
        m = E(model=self._m(), test_ecog=ecog_4x5x8, test_labels=labels_4x5x3)
        assert 0 <= m["phoneme_accuracy"] <= 1

    def test_all_keys(self, ecog_4x5x8, labels_4x5x3):
        from neurobridge.evaluate import evaluate_model as E
        m = E(model=self._m(), test_ecog=ecog_4x5x8, test_labels=labels_4x5x3)
        assert all(k in m for k in ["phoneme_accuracy","best_sample_accuracy","worst_sample_accuracy","num_test_samples","num_timesteps"])

    def test_count(self, ecog_4x5x8, labels_4x5x3):
        from neurobridge.evaluate import evaluate_model as E
        assert E(model=self._m(), test_ecog=ecog_4x5x8, test_labels=labels_4x5x3)["num_test_samples"] == 4

    def test_best_ge_worst(self, ecog_4x5x8, labels_4x5x3):
        from neurobridge.evaluate import evaluate_model as E
        m = E(model=self._m(), test_ecog=ecog_4x5x8, test_labels=labels_4x5x3)
        assert m["best_sample_accuracy"] >= m["worst_sample_accuracy"]

    def test_single_sample(self):
        from neurobridge.evaluate import evaluate_model as E
        e = mk_ecog(8, 5, seed=42).reshape(1, 5, 8); l = mk_labels(5, 3, seed=42).reshape(1, 5, 3)
        assert E(model=self._m(), test_ecog=e, test_labels=l)["num_test_samples"] == 1

    def test_accs_bounded(self):
        from neurobridge.evaluate import evaluate_model as E
        e = np.stack([mk_ecog(8, 5, seed=i) for i in range(10)])
        l = np.stack([mk_labels(5, 3, seed=i+100) for i in range(10)])
        m = E(model=self._m(), test_ecog=e, test_labels=l)
        assert all(0 <= m[k] <= 1 for k in ["phoneme_accuracy","best_sample_accuracy","worst_sample_accuracy"])

    def test_best_ge_mean(self):
        from neurobridge.evaluate import evaluate_model as E
        e = np.stack([mk_ecog(8, 5, seed=i) for i in range(8)])
        l = np.stack([mk_labels(5, 3, seed=i+200) for i in range(8)])
        m = E(model=self._m(), test_ecog=e, test_labels=l)
        assert m["best_sample_accuracy"] >= m["phoneme_accuracy"]

    def test_worst_le_mean(self):
        from neurobridge.evaluate import evaluate_model as E
        e = np.stack([mk_ecog(8, 5, seed=i) for i in range(8)])
        l = np.stack([mk_labels(5, 3, seed=i+300) for i in range(8)])
        m = E(model=self._m(), test_ecog=e, test_labels=l)
        assert m["worst_sample_accuracy"] <= m["phoneme_accuracy"]

    def test_timesteps_correct(self):
        from neurobridge.evaluate import evaluate_model as E
        e = mk_ecog(8, 5, seed=42).reshape(1, 5, 8); l = mk_labels(5, 3, seed=42).reshape(1, 5, 3)
        assert E(model=self._m(), test_ecog=e, test_labels=l)["num_timesteps"] == 5

    def test_no_labels_raises(self):
        from neurobridge.evaluate import evaluate_model as E
        with pytest.raises(ValueError): E(model=self._m(), test_ecog=mk_ecog(8,5).reshape(1,5,8), test_labels=None)

class TestDecode:
    def test_arpabet(self):
        from neurobridge.evaluate import decode_phoneme_sequence as D
        assert D(np.array([0,2,1]), phoneme_map=ARPABET) == ["AA","AH","AE"]

    def test_silence(self):
        from neurobridge.evaluate import decode_phoneme_sequence as D
        assert D(np.array([39,0,39]), phoneme_map=ARPABET) == ["SIL","AA","SIL"]

    def test_empty(self):
        from neurobridge.evaluate import decode_phoneme_sequence as D
        assert D(np.array([], dtype=int), phoneme_map=ARPABET) == []

    def test_full_arpabet(self):
        from neurobridge.evaluate import decode_phoneme_sequence as D
        assert D(np.arange(len(ARPABET)), phoneme_map=ARPABET) == ARPABET

    def test_repeated(self):
        from neurobridge.evaluate import decode_phoneme_sequence as D
        assert all(l == "AA" for l in D(np.array([0,0,0,0]), phoneme_map=ARPABET))

    def test_default_map(self):
        from neurobridge.evaluate import decode_phoneme_sequence as D
        assert D(np.array([0,1]))[0] == "PH_0"

    def test_load_missing(self):
        from neurobridge.evaluate import load_model as L; from pathlib import Path
        with pytest.raises(FileNotFoundError): L(Path("/nonexistent/m.keras"))

# ═══ INFERENCE (25 tests) ═══
class TestInference:
    def _d(self, f=16, c=5):
        from neurobridge.inference import RealtimeDecoder; from neurobridge.model import build_realtime_decoder
        d = RealtimeDecoder(); d.model = build_realtime_decoder(features=f, num_classes=c, lstm_units=16)
        d._features = f; d._num_phonemes = c; return d

    def test_init_none(self):
        from neurobridge.inference import RealtimeDecoder; assert RealtimeDecoder().model is None
    def test_no_model_raises(self):
        from neurobridge.inference import RealtimeDecoder
        with pytest.raises(RuntimeError): RealtimeDecoder().predict(np.zeros(128, dtype=np.float32))
    def test_wrong_shape(self):
        with pytest.raises(ValueError): self._d().predict(np.zeros((2,16), dtype=np.float32))
    def test_wrong_feat(self):
        with pytest.raises(ValueError): self._d().predict(np.zeros(32, dtype=np.float32))
    def test_shape(self, frame16): assert self._d().predict(frame16).shape == (5,)
    def test_sums(self, frame16): np.testing.assert_allclose(self._d().predict(frame16).sum(), 1.0, atol=1e-5)
    def test_label(self, frame16):
        l = self._d().predict_label(frame16, phoneme_map=ARPABET[:5])
        assert l in ARPABET[:5]
    def test_topk_count(self, frame16): assert len(self._d().predict_top_k(frame16, k=3, phoneme_map=ARPABET[:5])) == 3
    def test_topk_sorted(self, frame16):
        p = [x[1] for x in self._d().predict_top_k(frame16, k=5, phoneme_map=ARPABET[:5])]
        assert p == sorted(p, reverse=True)
    def test_topk_unique(self, frame16):
        ls = [x[0] for x in self._d().predict_top_k(frame16, k=5, phoneme_map=ARPABET[:5])]
        assert len(ls) == len(set(ls))
    def test_standalone_shape(self, frame8):
        from neurobridge.inference import predict_realtime_phoneme as P; from neurobridge.model import build_realtime_decoder as B
        assert P(frame8, B(features=8, num_classes=3, lstm_units=16)).shape == (3,)
    def test_standalone_sums(self, frame8):
        from neurobridge.inference import predict_realtime_phoneme as P; from neurobridge.model import build_realtime_decoder as B
        np.testing.assert_allclose(P(frame8, B(features=8, num_classes=3, lstm_units=16)).sum(), 1.0, atol=1e-5)
    def test_consistent(self):
        d = self._d()
        f = mk_ecog(16, 1, seed=42).flatten()
        p1 = d.predict(f)
        d.reset_state()
        p2 = d.predict(f)
        np.testing.assert_array_equal(p1, p2)
    def test_zeros(self):
        d = self._d(); p = d.predict(np.zeros(16, dtype=np.float32))
        np.testing.assert_allclose(p.sum(), 1.0, atol=1e-5)
    def test_ones(self):
        d = self._d(); p = d.predict(np.ones(16, dtype=np.float32))
        np.testing.assert_allclose(p.sum(), 1.0, atol=1e-5)
    def test_negative(self):
        d = self._d(); p = d.predict(np.full(16, -.5, dtype=np.float32))
        assert not np.any(np.isnan(p))
    def test_large(self):
        assert not np.any(np.isnan(self._d().predict(np.full(16, 500, dtype=np.float32))))
    def test_nonneg(self):
        assert np.all(self._d().predict(mk_ecog(16, 1, seed=99).flatten()) >= 0)
    def test_topk_over_classes(self):
        d = self._d(f=8, c=3)
        assert len(d.predict_top_k(mk_ecog(8,1,seed=42).flatten(), k=10, phoneme_map=ARPABET[:3])) == 3
    def test_topk_valid(self):
        d = self._d(); f = mk_ecog(16, 1, seed=42).flatten()
        for l, p in d.predict_top_k(f, k=5, phoneme_map=ARPABET[:5]):
            assert 0 <= p <= 1 and isinstance(l, str)
    def test_load_missing(self, tmp_path):
        from neurobridge.inference import RealtimeDecoder; from neurobridge.config import Config
        d = RealtimeDecoder(); orig = Config.MODEL_DIR
        Config.MODEL_DIR = tmp_path / "empty"; Config.MODEL_DIR.mkdir()
        try:
            with pytest.raises(FileNotFoundError): d.load()
        finally: Config.MODEL_DIR = orig
    def test_float64_cast(self):
        from neurobridge.inference import predict_realtime_phoneme as P; from neurobridge.model import build_realtime_decoder as B
        assert P(np.linspace(0,1,8).astype(np.float64), B(features=8, num_classes=3, lstm_units=16)).shape == (3,)
    def test_realistic_frame(self):
        from neurobridge.inference import predict_realtime_phoneme as P; from neurobridge.model import build_realtime_decoder as B
        p = P(mk_ecog(8, 1, seed=42).flatten(), B(features=8, num_classes=3, lstm_units=16))
        np.testing.assert_allclose(p.sum(), 1.0, atol=1e-5)

# ═══ SYNTHESIZER (8 tests) ═══
H = np.array([15,4,20,24], dtype=np.int32)  # HH AW L OW
Y = np.array([36,4,28], dtype=np.int32)      # Y AW S

class TestSynth:
    def test_array(self):
        from neurobridge.synthesizer import synthesize_speech_from_phonemes as S
        assert isinstance(S(H, sampling_rate=8000, phoneme_map=ARPABET), np.ndarray)
    def test_f32(self):
        from neurobridge.synthesizer import synthesize_speech_from_phonemes as S
        assert S(Y, sampling_rate=8000, phoneme_map=ARPABET).dtype == np.float32
    def test_len(self):
        from neurobridge.synthesizer import synthesize_speech_from_phonemes as S
        assert len(S(H, sampling_rate=8000, phoneme_duration_sec=.1, phoneme_map=ARPABET)) == 3200
    def test_nonsilent(self):
        from neurobridge.synthesizer import synthesize_speech_from_phonemes as S
        assert np.abs(S(H, sampling_rate=8000, phoneme_map=ARPABET)).max() > 0
    def test_single(self):
        from neurobridge.synthesizer import synthesize_speech_from_phonemes as S
        assert len(S(np.array([39], dtype=np.int32), sampling_rate=16000, phoneme_duration_sec=.05, phoneme_map=ARPABET)) == 800

class TestPhonemeText:
    def test_hello(self):
        from neurobridge.synthesizer import phoneme_ids_to_text as T
        assert T(H, phoneme_map=ARPABET) == "HH AW L OW"
    def test_yes(self):
        from neurobridge.synthesizer import phoneme_ids_to_text as T
        assert T(Y, phoneme_map=ARPABET) == "Y AW S"
    def test_sil(self):
        from neurobridge.synthesizer import phoneme_ids_to_text as T
        assert T(np.array([39], dtype=np.int32), phoneme_map=ARPABET) == "SIL"

# ═══ INTEGRATION (13 tests) ═══
class TestPipeline:
    def test_train_1_epoch(self):
        from neurobridge.train import train_model; from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.keras"; h = train_model(epochs=1, batch_size=4, train_samples=8, save_path=p)
            assert "loss" in h.history and p.exists()

    def test_loss_positive(self):
        from neurobridge.train import train_model; from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            assert train_model(epochs=1, batch_size=4, train_samples=8, save_path=Path(d)/"m.keras").history["loss"][0] > 0

    def test_acc_bounded(self):
        from neurobridge.train import train_model; from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            assert 0 <= train_model(epochs=1, batch_size=4, train_samples=8, save_path=Path(d)/"m.keras").history["accuracy"][0] <= 1

    def test_offline_e2e(self):
        from neurobridge.model import build_neurobridge_decoder as B, compile_model as C
        m = B(timesteps=5, features=8, num_classes=3, lstm_units=16); C(m)
        o = m.predict(mk_ecog(8, 5, seed=42).reshape(1, 5, 8), verbose=0)
        assert o.shape == (1, 5, 3)
        np.testing.assert_allclose(o.sum(axis=-1), 1.0, atol=1e-5)

    def test_rt_e2e(self):
        from neurobridge.model import build_realtime_decoder as B, compile_model as C
        m = B(features=16, num_classes=5, lstm_units=16); C(m)
        o = m.predict(mk_ecog(16, 1, seed=42).reshape(1, 1, 16), verbose=0)
        np.testing.assert_allclose(o.sum(), 1.0, atol=1e-5)

    def test_roundtrip(self):
        import tensorflow as tf; from neurobridge.model import build_neurobridge_decoder as B, compile_model as C; from pathlib import Path
        m = B(timesteps=5, features=8, num_classes=3, lstm_units=16); C(m)
        x = mk_ecog(8, 5, seed=42).reshape(1, 5, 8)
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d)/"rt.keras"); m.save(p)
            np.testing.assert_allclose(m.predict(x, verbose=0), tf.keras.models.load_model(p).predict(x, verbose=0), atol=1e-5)

    def test_config_phonemes_match(self):
        from neurobridge.config import Config; from neurobridge.model import build_neurobridge_decoder as B
        assert B().output_shape[-1] == Config.NUM_PHONEMES

    def test_config_features_match(self):
        from neurobridge.config import Config; from neurobridge.model import build_neurobridge_decoder as B
        assert B().input_shape[-1] == Config.NUM_FEATURES

    def test_config_timesteps_match(self):
        from neurobridge.config import Config; from neurobridge.model import build_neurobridge_decoder as B
        assert B().input_shape[-2] == Config.NUM_TIMESTEPS

    def test_decode_then_synth(self):
        from neurobridge.model import build_neurobridge_decoder as B
        from neurobridge.synthesizer import synthesize_speech_from_phonemes as S
        m = B(timesteps=5, features=8, num_classes=3, lstm_units=16)
        ids = np.argmax(m.predict(mk_ecog(8, 5, seed=42).reshape(1, 5, 8), verbose=0)[0], axis=-1)
        a = S(ids, sampling_rate=8000, phoneme_duration_sec=.1, phoneme_map=ARPABET[:3])
        assert a.dtype == np.float32 and len(a) == 4000

    def test_decode_then_text(self):
        from neurobridge.model import build_neurobridge_decoder as B
        from neurobridge.synthesizer import phoneme_ids_to_text as T
        m = B(timesteps=5, features=8, num_classes=3, lstm_units=16)
        ids = np.argmax(m.predict(mk_ecog(8, 5, seed=42).reshape(1, 5, 8), verbose=0)[0], axis=-1)
        assert len(T(ids, phoneme_map=ARPABET[:3]).split()) == 5

    def test_train_eval_e2e(self):
        from neurobridge.train import train_model; from neurobridge.evaluate import evaluate_model
        from neurobridge.config import Config; from pathlib import Path; import tensorflow as tf
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"m.keras"; train_model(epochs=1, batch_size=4, train_samples=8, save_path=p)
            m = tf.keras.models.load_model(str(p))
            e = np.stack([mk_ecog(Config.NUM_FEATURES, Config.NUM_TIMESTEPS, seed=i) for i in range(4)])
            l = np.stack([mk_labels(Config.NUM_TIMESTEPS, Config.NUM_PHONEMES, seed=i+500) for i in range(4)])
            assert 0 <= evaluate_model(model=m, test_ecog=e, test_labels=l)["phoneme_accuracy"] <= 1
