"""
NeuroBridge Decoder Architectures (class-based API)

Wraps the functional model builders from neurobridge.model into a
class-based interface used by the CLI, tuner, and legacy code paths.
Both APIs now produce identical model architectures.
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class NeurobridgeDecoder:
    """
    Phase 3: Core Neural Decoding Architectures for ECoG to Phoneme translation.
    Contains both the Offline (CNN + Bidirectional LSTM) and Online (CNN + Unidirectional LSTM) graphs.

    This class delegates to the functional builders in ``neurobridge.model``
    to guarantee that the class-based CLI path and the function-based test
    path always produce the exact same model architecture.
    """

    def __init__(
        self,
        timesteps: int = 100,
        channels: int = 128,
        phoneme_classes: int = 41,
        lstm_units: int = 256,
        dense_units: int = 128,
        dropout_rate: float = 0.3,
        cnn_filters: int = 64,
        cnn_kernel_size: int = 3,
    ):
        self.timesteps = timesteps
        self.channels = channels
        self.phoneme_classes = phoneme_classes
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size

    def build_offline_decoder(self):
        """
        Builds the context-rich CNN-Bidirectional LSTM for offline training.
        Input shape: (batch_size, timesteps, channels)
        """
        from neurobridge.model import build_neurobridge_decoder, compile_model

        logger.info(
            "Building Offline Decoder [CNN: %dx%d, LSTM: %d, Dense: %d, Dropout: %.1f]",
            self.cnn_filters,
            self.cnn_kernel_size,
            self.lstm_units,
            self.dense_units,
            self.dropout_rate,
        )
        model = build_neurobridge_decoder(
            timesteps=self.timesteps,
            features=self.channels,
            num_classes=self.phoneme_classes,
            lstm_units=self.lstm_units,
            dense_units=self.dense_units,
            dropout_rate=self.dropout_rate,
            cnn_filters=self.cnn_filters,
            cnn_kernel_size=self.cnn_kernel_size,
        )
        compile_model(model)
        return model

    def build_online_decoder(self):
        """
        Builds the low-latency CNN-Unidirectional LSTM for real-time inference.
        """
        from neurobridge.model import build_realtime_decoder

        logger.info(
            "Building Online Decoder [CNN: %dx%d, LSTM: %d, Dense: %d]",
            self.cnn_filters,
            self.cnn_kernel_size,
            self.lstm_units,
            self.dense_units,
        )
        model = build_realtime_decoder(
            features=self.channels,
            num_classes=self.phoneme_classes,
            lstm_units=self.lstm_units,
            dense_units=self.dense_units,
            cnn_filters=self.cnn_filters,
            cnn_kernel_size=self.cnn_kernel_size,
        )
        return model
