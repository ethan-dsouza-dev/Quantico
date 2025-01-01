import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error

def predictive_score_metrics(dataX, dataX_hat):
    # Basic Parameters
    No = len(dataX)
    data_dim = dataX[0].shape[1]
    
    # Maximum seq length and each seq length
    dataT = []
    Max_Seq_Len = 0
    for i in range(No):
        Max_Seq_Len = max(Max_Seq_Len, dataX[i].shape[0])
        dataT.append(dataX[i].shape[0])
    
    # Network Parameters
    hidden_dim = max(int(data_dim / 2), 1)
    iterations = 5000
    batch_size = 128
    
    # RNN predictor network
    class RNNPredictor(tf.keras.Model):
        def __init__(self, hidden_dim):
            super(RNNPredictor, self).__init__()
            self.gru = tf.keras.layers.GRU(
                hidden_dim, activation='tanh', return_sequences=True, return_state=False
            )
            self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        
        def call(self, inputs, sequence_lengths):
            rnn_outputs = self.gru(inputs, mask=tf.sequence_mask(sequence_lengths))
            outputs = self.output_layer(rnn_outputs)
            return outputs

    # Initialize the model
    predictor = RNNPredictor(hidden_dim)
    optimizer = tf.keras.optimizers.Adam()

    # Loss function
    def compute_loss(Y, Y_pred):
        return tf.reduce_mean(tf.abs(Y - Y_pred))

    # Training function
    @tf.function
    def train_step(X_mb, T_mb, Y_mb):
        with tf.GradientTape() as tape:
            Y_pred = predictor(X_mb, T_mb)
            loss = compute_loss(Y_mb, Y_pred)
        gradients = tape.gradient(loss, predictor.trainable_variables)
        optimizer.apply_gradients(zip(gradients, predictor.trainable_variables))
        return loss

    # Prepare the synthetic dataset for training
    for itt in range(iterations):
        idx = np.random.permutation(len(dataX_hat))
        train_idx = idx[:batch_size]

        X_mb_ = tf.ragged.constant([dataX_hat[i][:-1, :(data_dim - 1)] for i in train_idx])
        X_mb = X_mb_.to_tensor()
        T_mb = tf.constant([dataT[i] - 1 for i in train_idx], dtype=tf.int32)
        Y_mb = tf.ragged.constant([dataX_hat[i][1:, (data_dim - 1):] for i in train_idx]).to_tensor()

        loss = train_step(X_mb, T_mb, Y_mb)

    # Test on original dataset
    idx = np.random.permutation(len(dataX_hat))
    train_idx = idx[:No]

    X_mb_ = tf.ragged.constant([dataX[i][:-1, :(data_dim - 1)] for i in train_idx])
    X_mb = X_mb_.to_tensor()
    T_mb = tf.constant([dataT[i] - 1 for i in train_idx], dtype=tf.int32)
    Y_mb = tf.ragged.constant([dataX[i][1:, (data_dim - 1):] for i in train_idx]).to_tensor()

    pred_Y_curr = predictor(X_mb, T_mb)

    # Compute MAE
    MAE_Temp = 0
    for i in range(No):
        MAE_Temp += mean_absolute_error(Y_mb[i].numpy(), pred_Y_curr[i].numpy())

    MAE = MAE_Temp / No
    return MAE
