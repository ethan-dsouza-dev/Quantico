import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import keras

@tf.keras.utils.register_keras_serializable()
class Embedder(tf.keras.Model):
    hidden_dim = None
    num_layers = None

    def __init__(self, hidden_dim, num_layers, **kwargs):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        super(Embedder, self).__init__(**kwargs)
        self.rnn = Sequential([
            LSTM(hidden_dim, return_sequences=True) for _ in range(num_layers)
        ])
        self.dense = Dense(hidden_dim, activation='sigmoid')
    
    def call(self, X):
        H = self.rnn(X)
        return self.dense(H)

    def get_config(self):
        # Return all arguments needed to reconstruct the model
        base_config = super().get_config()
        config = {
            "rnn": self.rnn,
            "dense": self.dense,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        rnn_config = config.pop("rnn")
        rnn = keras.saving.deserialize_keras_object(rnn_config)
        dense_config = config.pop("dense")
        dense = keras.saving.deserialize_keras_object(dense_config)

        hidden_dim_config = config.pop("hidden_dim")
        hidden_dim = keras.saving.deserialize_keras_object(hidden_dim_config)
        num_layers_config = config.pop("num_layers")
        num_layers = keras.saving.deserialize_keras_object(num_layers_config)
        return cls(rnn, dense, hidden_dim, num_layers, **config)

@tf.keras.utils.register_keras_serializable()
class Recovery(tf.keras.Model):
    hidden_dim = None
    num_layers = None
    data_dim = None

    def __init__(self, data_dim, hidden_dim, num_layers, **kwargs):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.data_dim = data_dim

        super(Recovery, self).__init__(**kwargs)
        self.rnn = Sequential([
            LSTM(hidden_dim, return_sequences=True) for _ in range(num_layers)
        ])
        self.dense = Dense(data_dim, activation='sigmoid')
    
    def call(self, H):
        return self.dense(self.rnn(H))

    def get_config(self):
        # Return all arguments needed to reconstruct the model
        base_config = super().get_config()
        config = {
            "rnn": self.rnn,
            "dense": self.dense,
            "data_dim": self.data_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        rnn_config = config.pop("rnn")
        rnn = keras.saving.deserialize_keras_object(rnn_config)
        dense_config = config.pop("dense")
        dense = keras.saving.deserialize_keras_object(dense_config)

        hidden_dim_config = config.pop("hidden_dim")
        hidden_dim = keras.saving.deserialize_keras_object(hidden_dim_config)
        num_layers_config = config.pop("num_layers")
        num_layers = keras.saving.deserialize_keras_object(num_layers_config)
        data_dim_config = config.pop("data_dim")
        data_dim = keras.saving.deserialize_keras_object(data_dim_config)
        return cls(rnn, dense, data_dim, hidden_dim, num_layers, **config)

class Generator(tf.keras.Model):
    hidden_dim = None
    num_layers = None

    def __init__(self, hidden_dim, num_layers):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        super(Generator, self).__init__()
        self.rnn = Sequential([
            LSTM(hidden_dim, return_sequences=True) for _ in range(num_layers)
        ])
        self.dense = Dense(hidden_dim, activation='sigmoid')
    
    def call(self, Z):
        return self.dense(self.rnn(Z))

    def get_config(self):
        # Return all arguments needed to reconstruct the model
        base_config = super().get_config()
        config = {
            "rnn": self.rnn,
            "dense": self.dense,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        rnn_config = config.pop("rnn")
        rnn = keras.saving.deserialize_keras_object(rnn_config)
        dense_config = config.pop("dense")
        dense = keras.saving.deserialize_keras_object(dense_config)

        hidden_dim_config = config.pop("hidden_dim")
        hidden_dim = keras.saving.deserialize_keras_object(hidden_dim_config)
        num_layers_config = config.pop("num_layers")
        num_layers = keras.saving.deserialize_keras_object(num_layers_config)
        return cls(rnn, dense, hidden_dim, num_layers, **config)

class Supervisor(tf.keras.Model):
    hidden_dim = None
    num_layers = None

    def __init__(self, hidden_dim, num_layers):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        super(Supervisor, self).__init__()
        self.rnn = Sequential([
            LSTM(hidden_dim, return_sequences=True) for _ in range(num_layers - 1)
        ])
        self.dense = Dense(hidden_dim, activation='sigmoid')
    
    def call(self, H):
        return self.dense(self.rnn(H))

    def get_config(self):
        # Return all arguments needed to reconstruct the model
        base_config = super().get_config()
        config = {
            "rnn": self.rnn,
            "dense": self.dense,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        rnn_config = config.pop("rnn")
        rnn = keras.saving.deserialize_keras_object(rnn_config)
        dense_config = config.pop("dense")
        dense = keras.saving.deserialize_keras_object(dense_config)

        hidden_dim_config = config.pop("hidden_dim")
        hidden_dim = keras.saving.deserialize_keras_object(hidden_dim_config)
        num_layers_config = config.pop("num_layers")
        num_layers = keras.saving.deserialize_keras_object(num_layers_config)
        return cls(rnn, dense, hidden_dim, num_layers, **config)

class Discriminator(tf.keras.Model):
    hidden_dim = None
    num_layers = None

    def __init__(self, hidden_dim, num_layers):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        super(Discriminator, self).__init__()
        self.rnn = Sequential([
            LSTM(hidden_dim, return_sequences=True) for _ in range(num_layers)
        ])
        self.dense = Dense(1, activation=None)
    
    def call(self, H):
        return self.dense(self.rnn(H))

    def get_config(self):
        # Return all arguments needed to reconstruct the model
        base_config = super().get_config()
        config = {
            "rnn": self.rnn,
            "dense": self.dense,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        rnn_config = config.pop("rnn")
        rnn = keras.saving.deserialize_keras_object(rnn_config)
        dense_config = config.pop("dense")
        dense = keras.saving.deserialize_keras_object(dense_config)

        hidden_dim_config = config.pop("hidden_dim")
        hidden_dim = keras.saving.deserialize_keras_object(hidden_dim_config)
        num_layers_config = config.pop("num_layers")
        num_layers = keras.saving.deserialize_keras_object(num_layers_config)
        return cls(rnn, dense, hidden_dim, num_layers, **config)
