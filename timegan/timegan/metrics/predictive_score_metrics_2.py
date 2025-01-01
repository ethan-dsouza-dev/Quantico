import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

def predictive_score_metrics(dataX, dataX_hat):
    """
    dataX: list of [seq_len, data_dim] arrays (original data)
    dataX_hat: list of [seq_len, data_dim] arrays (synthetic data)
    
    Returns:
      MAE: Mean Absolute Error of one-step ahead prediction
    """
    # ------------------------------------------------------
    # Keep old variable names
    # ------------------------------------------------------
    # Basic Parameters
    No = len(dataX)
    data_dim = len(dataX[0][0, :])
    
    # Maximum seq length and each seq length
    dataT = []
    Max_Seq_Len = 0
    for i in range(No):
        seq_len = len(dataX[i][:, 0])
        Max_Seq_Len = max(Max_Seq_Len, seq_len)
        dataT.append(seq_len)
    
    # Network Parameters
    hidden_dim = max(int(data_dim / 2), 1)
    iterations = 1000
    batch_size = 128
    
    # ------------------------------------------------------
    # Define "gru_layer" and "dense_layer" as in old code
    # ------------------------------------------------------
    gru_layer = tf.keras.layers.GRU(
        units=hidden_dim,
        activation='tanh',
        return_sequences=True,
        name='gru_layer'  # optional name
    )
    dense_layer = tf.keras.layers.Dense(1, activation=None, name='dense_layer')
    
    # ------------------------------------------------------
    # predictor function (same variable names)
    # ------------------------------------------------------
    def predictor(X, T):
        """
        X: Tensor of shape [batch_size, seq_len-1, data_dim-1]
        T: 1D int tensor [batch_size] with each seq length-1
        """
        mask = tf.sequence_mask(T, maxlen=tf.shape(X)[1])
        d_outputs = gru_layer(X, mask=mask)     # shape: [batch_size, seq_len-1, hidden_dim]
        Y_hat = dense_layer(d_outputs)          # shape: [batch_size, seq_len-1, 1]
        Y_hat_Final = tf.nn.sigmoid(Y_hat)      # Sigmoid output
        return Y_hat_Final
    
    # d_vars = trainable variables of the two layers
    X_dummy = tf.zeros([1, 1, data_dim - 1], dtype=tf.float32)
    T_dummy = tf.constant([1], dtype=tf.int32)
    _ = predictor(X_dummy, T_dummy)  # <-- Dummy pass here

    d_vars = gru_layer.trainable_variables + dense_layer.trainable_variables
    
    # ------------------------------------------------------
    # Create an optimizer named "D_solver" to match old code
    # ------------------------------------------------------
    D_solver = tf.keras.optimizers.Adam()
    
    # We'll define a small function that computes "D_loss" for a given (X, Y, T)
    # We'll keep the name "D_loss" as in the old code, but we'll compute it on the fly.
    def compute_D_loss(X_batch, Y_batch, T_batch):
        # Forward pass
        Y_pred = predictor(X_batch, T_batch)
        # Using mean absolute difference, similar to TF1's tf.compat.v1.losses.absolute_difference
        return tf.reduce_mean(tf.abs(Y_batch - Y_pred))
    
    # ------------------------------------------------------
    # Helper function to get a training batch from dataX_hat
    # (similar to how old code does slicing, but we pad for TF2)
    # ------------------------------------------------------
    def get_batch(dataX_, dataT_, b_size):
        # dataX_ is dataX_hat or dataX
        idx = np.random.permutation(len(dataX_))[:b_size]
        
        X_mb_list = []
        T_mb_list = []
        Y_mb_list = []
        
        for i in idx:
            # For each sample, shape = [seq_len, data_dim]
            # We'll do one-step-ahead: X: [seq_len-1, data_dim-1], Y: [seq_len-1, 1]
            seq_len = dataX_[i].shape[0]  # full length
            X_mb_list.append(dataX_[i][:-1, :data_dim-1])  # up to second last step, columns 0..(data_dim-2)
            Y_mb_list.append(np.expand_dims(dataX_[i][1:, data_dim-1], axis=-1))  # steps 1..end, last column
            T_mb_list.append(seq_len - 1)  # each sample now has length (seq_len - 1)
        
        # Pad them to same length in this batch
        max_len = max(T_mb_list)
        # Build arrays
        X_mb = np.zeros((b_size, max_len, data_dim-1), dtype=np.float32)
        Y_mb = np.zeros((b_size, max_len, 1), dtype=np.float32)
        
        for j in range(b_size):
            slen = T_mb_list[j]
            X_mb[j, :slen, :] = X_mb_list[j]
            Y_mb[j, :slen, :] = Y_mb_list[j]
        
        return X_mb, np.array(T_mb_list, dtype=np.int32), Y_mb
    
    # ------------------------------------------------------
    # 1) Training loop using Synthetic dataset
    # (Matches old code logic with "iterations" and "batch_size")
    # ------------------------------------------------------
    #print(range(iterations))
    for itt in range(iterations):
        X_mb_np, T_mb_np, Y_mb_np = get_batch(dataX_hat, dataT, batch_size)
        
        # Convert to tensors
        X_mb_tf = tf.constant(X_mb_np, dtype=tf.float32)
        T_mb_tf = tf.constant(T_mb_np, dtype=tf.int32)
        Y_mb_tf = tf.constant(Y_mb_np, dtype=tf.float32)
        
        # GradientTape for TF2
        with tf.GradientTape() as tape:
            step_d_loss = compute_D_loss(X_mb_tf, Y_mb_tf, T_mb_tf)
        
        grads = tape.gradient(step_d_loss, d_vars)
        D_solver.apply_gradients(zip(grads, d_vars))
        
        # (Optional) print every 1000 steps
        if itt % 1000 == 0:
            print(f"step: {itt}, D_loss: {step_d_loss.numpy():.4f}")
    
    print("loop done")
    # ------------------------------------------------------
    # 2) Test on the Original dataset to compute final MAE
    # (like old code does at the end)
    # ------------------------------------------------------
    # We'll build a "batch" that includes all original samples at once
    # in typical code, we'd do a loop if memory is large, but let's do it directly.
    
    # Instead of random, we take the entire set
    X_mb_list = []
    T_mb_list = []
    Y_mb_list = []
    print("before for loop")
    for i in range(No):
        seq_len = dataX[i].shape[0]
        X_mb_list.append(dataX[i][:-1, :data_dim-1])
        Y_mb_list.append(np.expand_dims(dataX[i][1:, data_dim-1], axis=-1))
        T_mb_list.append(seq_len - 1)
    
    print("get past for loop")
    max_len_test = max(T_mb_list)
    X_mb_test = np.zeros((No, max_len_test, data_dim-1), dtype=np.float32)
    Y_mb_test = np.zeros((No, max_len_test, 1), dtype=np.float32)
    
    print("before for loop 2")
    for j in range(No):
        slen = T_mb_list[j]
        X_mb_test[j, :slen, :] = X_mb_list[j]
        Y_mb_test[j, :slen, :] = Y_mb_list[j]

    print("after for loop 2")
    
    T_mb_test = np.array(T_mb_list, dtype=np.int32)
    
    # Convert to TF
    X_mb_test_tf = tf.constant(X_mb_test, dtype=tf.float32)
    T_mb_test_tf = tf.constant(T_mb_test, dtype=tf.int32)
    # We'll get predicted Y
    pred_Y_curr = predictor(X_mb_test_tf, T_mb_test_tf).numpy()  # shape [No, max_len_test, 1]
    
    # Compute MAE for each sequence
    MAE_Temp = 0.0
    for i in range(No):
        slen = T_mb_list[i]
        y_true = Y_mb_test[i, :slen, 0]
        y_pred = pred_Y_curr[i, :slen, 0]
        MAE_Temp += mean_absolute_error(y_true, y_pred)
    
    MAE = MAE_Temp / No
    
    return MAE