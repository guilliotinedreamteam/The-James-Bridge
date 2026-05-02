import tensorflow as tf

class CategoricalFocalLoss(tf.keras.losses.Loss):
    """
    Categorical Focal Loss for extreme class imbalance in sequence-to-sequence models.
    Penalizes easy examples (majority class/silence) and focuses gradients on hard, rare phonemes.
    """
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate focal weight: alpha * (1 - p_t)^gamma
        weight = self.alpha * tf.math.pow((1.0 - y_pred), self.gamma)
        
        # Apply focal weight to cross entropy
        loss = weight * cross_entropy
        
        # Sum over classes, average over batch and timesteps
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
