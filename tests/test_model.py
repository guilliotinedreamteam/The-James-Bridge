"""Model architecture tests — 38 tests."""
import numpy as np, pytest, tempfile, os

# ── Build & Output ──
class TestOfflineDecoder:
    def test_builds(self):
        from neurobridge.model import build_neurobridge_decoder as B
        assert B(timesteps=5, features=8, num_classes=3) is not None

    def test_output_shape(self, ecog_2x5x8):
        from neurobridge.model import build_neurobridge_decoder as B
        assert B(timesteps=5, features=8, num_classes=3).predict(ecog_2x5x8, verbose=0).shape == (2, 5, 3)

    def test_softmax(self, ecog_1x5x8):
        from neurobridge.model import build_neurobridge_decoder as B
        np.testing.assert_allclose(B(timesteps=5, features=8, num_classes=3).predict(ecog_1x5x8, verbose=0).sum(axis=-1), 1.0, atol=1e-5)

    def test_compiles(self):
        from neurobridge.model import build_neurobridge_decoder as B, compile_model as C
        m = B(timesteps=5, features=8, num_classes=3); C(m)
        assert m.optimizer and m.loss

    def test_params_positive(self):
        from neurobridge.model import build_neurobridge_decoder as B
        assert B(timesteps=5, features=8, num_classes=3).count_params() > 0

class TestRealtimeDecoder:
    def test_builds(self):
        from neurobridge.model import build_realtime_decoder as B
        assert B(features=16, num_classes=5) is not None

    def test_output_shape(self, rt_frame16):
        from neurobridge.model import build_realtime_decoder as B
        assert B(features=16, num_classes=5).predict(rt_frame16, verbose=0).shape == (1, 5)

    def test_softmax(self, rt_frame16):
        from neurobridge.model import build_realtime_decoder as B
        np.testing.assert_allclose(B(features=16, num_classes=5).predict(rt_frame16, verbose=0).sum(), 1.0, atol=1e-5)

# ── Layer Structure ──
class TestLayers:
    def _m(self):
        from neurobridge.model import build_neurobridge_decoder as B
        return B(timesteps=5, features=8, num_classes=3)

    def test_has_bilstm(self):
        assert any("bi_lstm" in l.name for l in self._m().layers)

    def test_has_batchnorm(self):
        assert any("bn" in l.name for l in self._m().layers)

    def test_has_td(self):
        assert any("td_" in l.name for l in self._m().layers)

    def test_min_layers(self):
        assert len(self._m().layers) >= 6

    def test_name_offline(self):
        assert self._m().name == "neurobridge_decoder"

    def test_rt_no_bidir(self):
        from neurobridge.model import build_realtime_decoder as B
        assert not any("bidirectional" in l.name.lower() for l in B(features=8, num_classes=3).layers)

    def test_rt_name(self):
        from neurobridge.model import build_realtime_decoder as B
        assert B(features=8, num_classes=3).name == "neurobridge_realtime_decoder"

# ── Configurations ──
class TestConfigs:
    def test_small(self):
        from neurobridge.model import build_neurobridge_decoder as B
        assert B(timesteps=2, features=4, num_classes=2, lstm_units=16, dense_units=8).count_params() > 0

    def test_100_classes(self):
        from neurobridge.model import build_neurobridge_decoder as B
        x = np.ones((1, 3, 4), dtype=np.float32) * .5
        assert B(timesteps=3, features=4, num_classes=100, lstm_units=16, dense_units=8).predict(x, verbose=0).shape == (1, 3, 100)

    def test_1_timestep(self):
        from neurobridge.model import build_neurobridge_decoder as B
        x = np.ones((1, 1, 4), dtype=np.float32) * .5
        assert B(timesteps=1, features=4, num_classes=3, lstm_units=16, dense_units=8).predict(x, verbose=0).shape == (1, 1, 3)

    def test_1_feature(self):
        from neurobridge.model import build_neurobridge_decoder as B
        x = np.ones((1, 3, 1), dtype=np.float32) * .5
        assert B(timesteps=3, features=1, num_classes=3, lstm_units=16, dense_units=8).predict(x, verbose=0).shape == (1, 3, 3)

    def test_more_units_more_params(self):
        from neurobridge.model import build_neurobridge_decoder as B
        assert B(timesteps=3, features=4, num_classes=3, lstm_units=64).count_params() > B(timesteps=3, features=4, num_classes=3, lstm_units=32).count_params()

# ── Determinism ──
class TestDeterminism:
    def test_same_in_same_out(self):
        from neurobridge.model import build_neurobridge_decoder as B
        m = B(timesteps=3, features=4, num_classes=3, lstm_units=16)
        x = np.ones((1, 3, 4), dtype=np.float32) * .5
        np.testing.assert_array_equal(m.predict(x, verbose=0), m.predict(x, verbose=0))

    def test_diff_in_diff_out(self, ecog_2x5x8):
        from neurobridge.model import build_neurobridge_decoder as B
        o = B(timesteps=5, features=8, num_classes=3, lstm_units=16).predict(ecog_2x5x8, verbose=0)
        assert not np.array_equal(o[0], o[1])

    def test_batch_vs_single(self, ecog_2x5x8):
        from neurobridge.model import build_neurobridge_decoder as B
        m = B(timesteps=5, features=8, num_classes=3, lstm_units=16)
        np.testing.assert_allclose(m.predict(ecog_2x5x8, verbose=0)[0], m.predict(ecog_2x5x8[0:1], verbose=0)[0], atol=1e-5)

# ── Serialization ──
class TestSerialization:
    def test_save_load_offline(self):
        import tensorflow as tf
        from neurobridge.model import build_neurobridge_decoder as B
        m = B(timesteps=3, features=4, num_classes=3, lstm_units=16)
        x = np.ones((1, 3, 4), dtype=np.float32) * .5
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.keras"); m.save(p)
            np.testing.assert_allclose(m.predict(x, verbose=0), tf.keras.models.load_model(p).predict(x, verbose=0), atol=1e-5)

    def test_save_load_realtime(self):
        import tensorflow as tf
        from neurobridge.model import build_realtime_decoder as B
        m = B(features=8, num_classes=3, lstm_units=16)
        x = np.ones((1, 1, 8), dtype=np.float32) * .5
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "rt.keras"); m.save(p)
            np.testing.assert_allclose(m.predict(x, verbose=0), tf.keras.models.load_model(p).predict(x, verbose=0), atol=1e-5)

# ── Compile ──
class TestCompile:
    def test_custom_lr(self):
        from neurobridge.model import build_neurobridge_decoder as B, compile_model as C
        m = B(timesteps=3, features=4, num_classes=3, lstm_units=16); C(m, learning_rate=0.01)
        assert m.optimizer is not None

    def test_loss_type(self):
        from neurobridge.model import build_neurobridge_decoder as B, compile_model as C
        m = B(timesteps=3, features=4, num_classes=3, lstm_units=16); C(m)
        assert "crossentropy" in str(m.loss).lower()

# ── Edge Cases ──
class TestEdgeCases:
    def _m(self):
        from neurobridge.model import build_neurobridge_decoder as B
        return B(timesteps=3, features=4, num_classes=3, lstm_units=16)

    def test_zeros(self):
        o = self._m().predict(np.zeros((1, 3, 4), dtype=np.float32), verbose=0)
        np.testing.assert_allclose(o.sum(axis=-1), 1.0, atol=1e-5)

    def test_ones(self):
        o = self._m().predict(np.ones((1, 3, 4), dtype=np.float32), verbose=0)
        np.testing.assert_allclose(o.sum(axis=-1), 1.0, atol=1e-5)

    def test_large_vals(self):
        assert not np.any(np.isnan(self._m().predict(np.ones((1, 3, 4), dtype=np.float32) * 1000, verbose=0)))

    def test_negative(self):
        o = self._m().predict(np.full((1, 3, 4), -1.0, dtype=np.float32), verbose=0)
        assert not np.any(np.isnan(o))
        np.testing.assert_allclose(o.sum(axis=-1), 1.0, atol=1e-5)

    def test_big_batch(self):
        assert self._m().predict(np.ones((64, 3, 4), dtype=np.float32) * .5, verbose=0).shape == (64, 3, 3)
