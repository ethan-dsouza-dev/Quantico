import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class Embedder(tf.keras.Model):
    def __init__(self, hidden_dim, num_layers):
        super(Embedder, self).__init__()
        self.rnn = Sequential([
            LSTM(hidden_dim, return_sequences=True) for _ in range(num_layers)
        ])
        self.dense = Dense(hidden_dim, activation='sigmoid')
    
    def call(self, X):
        H = self.rnn(X)
        return self.dense(H)

class Recovery(tf.keras.Model):
    def __init__(self, data_dim, hidden_dim, num_layers):
        super(Recovery, self).__init__()
        self.rnn = Sequential([
            LSTM(hidden_dim, return_sequences=True) for _ in range(num_layers)
        ])
        self.dense = Dense(data_dim, activation='sigmoid')
    
    def call(self, H):
        return self.dense(self.rnn(H))

class Generator(tf.keras.Model):

    def __init__(self, hidden_dim, num_layers):
        super(Generator, self).__init__()
        self.rnn = Sequential([
            LSTM(hidden_dim, return_sequences=True) for _ in range(num_layers)
        ])
        self.dense = Dense(hidden_dim, activation='sigmoid')
    
    def call(self, Z):
        return self.dense(self.rnn(Z))

class Supervisor(tf.keras.Model):

    def __init__(self, hidden_dim, num_layers):
        super(Supervisor, self).__init__()
        self.rnn = Sequential([
            LSTM(hidden_dim, return_sequences=True) for _ in range(num_layers - 1)
        ])
        self.dense = Dense(hidden_dim, activation='sigmoid')
    
    def call(self, H):
        return self.dense(self.rnn(H))

class Discriminator(tf.keras.Model):
    def __init__(self, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.rnn = Sequential([
            LSTM(hidden_dim, return_sequences=True) for _ in range(num_layers)
        ])
        self.dense = Dense(1, activation=None)
    
    def call(self, H):
        return self.dense(self.rnn(H))