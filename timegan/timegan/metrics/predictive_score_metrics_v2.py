import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error

#%% Post-hoc RNN one-step ahead predictor
def predictive_score_metrics(dataX, dataX_hat):
    # Basic Parameters
    No = len(dataX)
    data_dim = dataX[0].shape[1]

    # Maximum seq length and each seq length
    dataT = [len(seq) for seq in dataX]
    Max_Seq_Len = max(dataT)

    # Network Parameters
    hidden_dim = max(int(data_dim / 2), 1)
    iterations = 5000
    batch_size = 128

    #%% Build RNN Predictor Network
    class Predictor(tf.keras.Model):
        def __init__(self, hidden_dim):
            super(Predictor, self).__init__()
            self.rnn = tf.keras.layers.GRU(hidden_dim, activation='tanh', return_sequences=True)
            self.fc = tf.keras.layers.Dense(1, activation=None)

        def call(self, inputs):
            x, lengths = inputs
            x = self.rnn(x)
            x = self.fc(x)
            return tf.nn.sigmoid(x)

    predictor = Predictor(hidden_dim)

    # Optimizer and Loss
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanAbsoluteError()

    #%% Training Loop
    @tf.function
    def train_step(X_mb, T_mb, Y_mb):
        with tf.GradientTape() as tape:
            Y_pred = predictor((X_mb, T_mb))
            loss = loss_fn(Y_mb, Y_pred)
        gradients = tape.gradient(loss, predictor.trainable_variables)
        optimizer.apply_gradients(zip(gradients, predictor.trainable_variables))
        return loss

    # Training using Synthetic dataset
    for itt in range(iterations):
        idx = np.random.permutation(len(dataX_hat))[:batch_size]
        print(type(dataX_hat))
        X_mb = tf.ragged.constant([dataX_hat[i][:-1, :-1] for i in idx], dtype=tf.float32).to_tensor()
        T_mb = tf.constant([dataT[i] - 1 for i in idx], dtype=tf.int32)
        Y_mb = tf.ragged.constant([
            np.reshape(dataX_hat[i][1:, -1], (-1, 1)) for i in idx
        ], dtype=tf.float32).to_tensor()

        train_step(X_mb, T_mb, Y_mb)

    #%% Use Original Dataset to Test
    idx = np.arange(No)
    X_mb = tf.ragged.constant([dataX[i][:-1, :-1] for i in idx], dtype=tf.float32).to_tensor()
    T_mb = tf.constant([dataT[i] - 1 for i in idx], dtype=tf.int32)
    Y_mb = tf.ragged.constant([
        np.reshape(dataX[i][1:, -1], (-1, 1)) for i in idx
    ], dtype=tf.float32).to_tensor()

    pred_Y_curr = predictor((X_mb, T_mb))

    # Compute MAE
    MAE_Temp = 0
    for i in range(No):
        MAE_Temp += mean_absolute_error(Y_mb[i].numpy(), pred_Y_curr[i].numpy())

    MAE = MAE_Temp / No

    return MAE
