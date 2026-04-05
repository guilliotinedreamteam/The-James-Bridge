import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, BatchNormalization, Input, Dropout
except ImportError:
    logger.error("TensorFlow is not installed. Phase 3 cannot execute without it. Run 'pip install tensorflow'.")
    tf = None

class NeurobridgeDecoder:
    """
    Phase 3: Core Neural Decoding Architectures for ECoG to Phoneme translation.
    Contains both the Offline (Bidirectional) and Online (Unidirectional) LSTM graphs.
    """
    def __init__(self, timesteps: int = 100, channels: int = 128, phoneme_classes: int = 41,
                 lstm_units: int = 256, dense_units: int = 128, dropout_rate: float = 0.3):
        self.timesteps = timesteps
        self.channels = channels
        self.phoneme_classes = phoneme_classes
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        if tf is None:
            raise ImportError("TensorFlow is missing.")

    def build_offline_decoder(self) -> tf.keras.Model:
        """
        Builds the context-rich Bidirectional LSTM for offline training.
        Input shape: (batch_size, timesteps, channels)
        """
        logger.info(f"Building Offline Decoder [LSTM: {self.lstm_units}, Dense: {self.dense_units}, Dropout: {self.dropout_rate}]")
        
        model = Sequential([
            Input(shape=(self.timesteps, self.channels)),
            Bidirectional(LSTM(self.lstm_units, return_sequences=True)),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            TimeDistributed(Dense(self.dense_units, activation='relu')),
            Dropout(self.dropout_rate),
            TimeDistributed(Dense(self.phoneme_classes, activation='softmax'))
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def build_online_decoder(self) -> tf.keras.Model:
        """
        Builds the low-latency Unidirectional LSTM for real-time inference.
        """
        logger.info(f"Building Online Decoder [LSTM: {self.lstm_units}, Dense: {self.dense_units}]")
        
        model = Sequential([
            Input(shape=(1, self.channels)),
            LSTM(self.lstm_units, return_sequences=True), 
            BatchNormalization(),
            TimeDistributed(Dense(self.dense_units, activation='relu')),
            TimeDistributed(Dense(self.phoneme_classes, activation='softmax'))
        ])
        return model
