import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class SimpleAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(SimpleAttention, self).build(input_shape)

    def call(self, x):
        # Score calculation: e_t = tanh(x^T * W + b)
        e = tf.tanh(tf.matmul(x, self.W) + self.b)  # Shape: (batch_size, seq_len, 1)
        # Softmax pour obtenir les poids d'alignement
        a = tf.nn.softmax(e, axis=1)  # Shape: (batch_size, seq_len, 1)
        # Context vector: somme pondérée de x
        context_vector = tf.reduce_sum(a * x, axis=1)  # Shape: (batch_size, hidden_dim)
        return context_vector, a

# Modèle complet: GRU + Attention
inputs = keras.Input(shape=(None, 10))  # Exemple: seq_len variable, hidden_dim=10
gru_out = layers.GRU(64, return_sequences=True)(inputs)
context_vector, alignment_weights = SimpleAttention()(gru_out)
outputs = layers.Dense(1)(context_vector)
model = keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')

# Dataset synthétique: combinaison de sinusoïdes
def generate_data(seq_len=100):
    t = np.linspace(0, 20, seq_len)
    return np.stack([np.sin(t), np.sin(0.5*t), np.sin(2*t)], axis=-1)

# Modèle Seq2Seq avec Attention
encoder_inputs = keras.Input(shape=(None, 3))
encoder = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoder_inputs)

decoder_inputs = keras.Input(shape=(None, 1))
decoder_lstm = layers.LSTM(64, return_sequences=True)(decoder_inputs, initial_state=encoder.states)

# Cross-Attention: utilise les états de l'encodeur comme K,V
attention = layers.Attention()([decoder_lstm, encoder])
decoder_outputs = layers.Dense(1)(attention)
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# MLflow: Tracking de l'"Attention Span"
import mlflow
mlflow.set_experiment("LSTM_Attention_TimeSeries")
with mlflow.start_run():
    model.fit(...)
    # Log de la moyenne des poids d'attention non-nuls
    mlflow.log_metric("attention_span", tf.reduce_mean(tf.cast(alignment_weights > 0.1, tf.float32)).numpy())

# Exemple simplifié
class HierarchicalTAP(keras.Model):
    def __init__(self):
        super().__init__()
        self.slow_lstm = layers.LSTM(128, return_sequences=True)
        self.fast_lstm = layers.LSTM(64, return_sequences=True)
        self.gate = layers.Dense(1, activation='sigmoid')

    def call(self, x):
        slow = self.slow_lstm(x[:, ::5, :])  # Sous-échantillonnage temporel
        fast = self.fast_lstm(x)
        gate = self.gate(x)
        return gate * slow + (1 - gate) * fast
