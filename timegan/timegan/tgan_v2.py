# Implementation of tgan.py with up to date libraries

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
print(tf.__version__)
print(np.__version__)

# Min Max Normalizer
def MinMaxScaler(dataX):
    
    min_val = np.min(np.min(dataX, axis = 0), axis = 0)
    dataX = dataX - min_val
    
    max_val = np.max(np.max(dataX, axis = 0), axis = 0)
    dataX = dataX / (max_val + 1e-7)
    
    return dataX, min_val, max_val

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

def tgan(dataX, parameters):
    # Basic Parameters
    No = len(dataX)
    data_dim = len(dataX[0][0,:])

    # Maximum seq length and each seq length
    dataT = list()
    Max_Seq_Len = 0
    for i in range(No):
        Max_Seq_Len = max(Max_Seq_Len, len(dataX[i][:,0]))
        dataT.append(len(dataX[i][:,0]))
        
    # Normalization
    if ((np.max(dataX) > 1) | (np.min(dataX) < 0)):
        dataX, min_val, max_val = MinMaxScaler(dataX)
        Normalization_Flag = 1
    else:
        Normalization_Flag = 0
    
    dataXnp = np.array(dataX)
    dataTnp = np.array(dataT)
     
    # Network Parameters
    hidden_dim   = parameters['hidden_dim'] 
    num_layers   = parameters['num_layers']
    iterations   = parameters['iterations']
    batch_size   = parameters['batch_size']
    module_name  = parameters['module_name']    # 'lstm' or 'lstmLN'
    z_dim        = parameters['z_dim']
    gamma        = 1

    # X = tf.placeholder(tf.float32, [None, Max_Seq_Len, data_dim], name = "myinput_x")
    # Z = tf.placeholder(tf.float32, [None, Max_Seq_Len, z_dim], name = "myinput_z")
    # T = tf.placeholder(tf.int32, [None], name = "myinput_t")
    # X = tf.random.normal([batch_size, Max_Seq_Len, data_dim])  # Simulated input data
    # Z = tf.random.normal([batch_size, Max_Seq_Len, z_dim])     # Random noise for generator
    # T = tf.random.uniform([batch_size], maxval=Max_Seq_Len, dtype=tf.int32)  # Sequence lengths

    # Network Initialization
    embedder = Embedder(hidden_dim=hidden_dim, num_layers=num_layers)
    recovery = Recovery(data_dim=data_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    generator = Generator(hidden_dim=hidden_dim, num_layers=num_layers)
    supervisor = Supervisor(hidden_dim=hidden_dim, num_layers=num_layers)
    discriminator = Discriminator(hidden_dim=hidden_dim, num_layers=num_layers)


    # Loss functions
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mse = tf.keras.losses.MeanSquaredError()

    # Optimizers
    # TODO: what is the lr here
    optimizer_e = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer_d = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer_g = tf.keras.optimizers.Adam(learning_rate=0.001)

    @tf.function
    def train_embedder(X_mb, T_mb, embedder, recovery):
        with tf.GradientTape() as tape:
            H = embedder(X_mb)
            X_tilde = recovery(H)
            E_loss_T0 = mse(X_mb, X_tilde)
            E_loss0 = 10 * tf.sqrt(E_loss_T0)
        gradients = tape.gradient(E_loss0, embedder.trainable_variables + recovery.trainable_variables)
        optimizer_e.apply_gradients(zip(gradients, embedder.trainable_variables + recovery.trainable_variables))
        return E_loss0

    @tf.function
    def train_generator(Z_mb, H_mb, T_mb, generator, supervisor, discriminator):
        with tf.GradientTape() as tape:
            H_hat = generator(Z_mb)
            H_hat_supervise = supervisor(H_hat)
            G_loss_U = bce(tf.ones_like(discriminator(H_hat)), discriminator(H_hat))
            G_loss_S = mse(H_mb[:, 1:, :], H_hat_supervise[:, 1:, :])
            G_loss = G_loss_U + gamma * G_loss_S
        gradients = tape.gradient(G_loss, generator.trainable_variables + supervisor.trainable_variables)
        optimizer_g.apply_gradients(zip(gradients, generator.trainable_variables + supervisor.trainable_variables))
        return G_loss
    
    @tf.function
    def train_discriminator(X_mb, Z_mb, T_mb, embedder, generator, discriminator):
        with tf.GradientTape() as tape:
            H = embedder(X_mb)
            H_hat = generator(Z_mb)
            Y_real = discriminator(H)
            Y_fake = discriminator(H_hat)
            D_loss_real = bce(tf.ones_like(Y_real), Y_real)
            D_loss_fake = bce(tf.zeros_like(Y_fake), Y_fake)
            D_loss = D_loss_real + D_loss_fake
        gradients = tape.gradient(D_loss, discriminator.trainable_variables)
        optimizer_d.apply_gradients(zip(gradients, discriminator.trainable_variables))
        return D_loss
    
    # Create a dataset once, outside the loop
    dataset = tf.data.Dataset.from_tensor_slices((dataXnp, dataTnp))

    # Shuffle, batch, and prefetch
    dataset = (
        tf.data.Dataset.from_tensor_slices((dataXnp, dataTnp))
        .shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        .repeat()  # Enables the dataset to be infinite
    )

    # import time
    # Embedder Training
    for itt, (X_mb, T_mb) in enumerate(dataset.take(iterations)):
    # for itt in range(iterations):
        # Select batch
        # idx = np.random.choice(No, batch_size, replace=False)
        # # print(type(dataXnp[idx]), dataXnp[idx].shape)
        # dataXnp = dataXnp.astype(np.float32)
        # X_mb = tf.convert_to_tensor(dataXnp[idx])
        # T_mb = tf.convert_to_tensor(dataTnp[idx])
        X_mb = tf.cast(X_mb, dtype=tf.float32)
        T_mb = tf.cast(T_mb, dtype=tf.int32)
        
        # Train embedder
        step_e_loss = train_embedder(X_mb, T_mb, embedder, recovery)

        if itt % 1000 == 0:
            print(f"Iter: {itt}, E_loss: {step_e_loss.numpy():.4f}")
    
    # Just a lambda function to generate random noise for batch_size
    Z_mb = lambda batch_size: tf.random.uniform((batch_size, Max_Seq_Len, z_dim))

    # Supervised Training
    for itt, (X_mb, T_mb) in enumerate(dataset.take(iterations)):
    # for itt in range(iterations):
        # Select batch
        # idx = np.random.choice(No, batch_size, replace=False)
        # X_mb = tf.convert_to_tensor(dataX[idx])
        # T_mb = tf.convert_to_tensor(dataT[idx])
        X_mb = tf.cast(X_mb, dtype=tf.float32)
        T_mb = tf.cast(T_mb, dtype=tf.int32)
        Z_batch = Z_mb(batch_size)
        
        # Train embedder
        step_g_loss = train_generator(Z_batch, embedder(X_mb), T_mb, generator, supervisor, discriminator)

        if itt % 1000 == 0:
            print(f"Iter: {itt}, G_loss: {step_g_loss.numpy():.4f}")

    # Joint Training
    for itt, (X_mb, T_mb) in enumerate(dataset.take(iterations)):
    # for itt in range(iterations):
        # Select batch
        # idx = np.random.choice(No, batch_size, replace=False)
        # X_mb = tf.convert_to_tensor(dataX[idx])
        # T_mb = tf.convert_to_tensor(dataT[idx])
        X_mb = tf.cast(X_mb, dtype=tf.float32)
        T_mb = tf.cast(T_mb, dtype=tf.int32)
        Z_batch = Z_mb(batch_size)
        
        # Train embedder
        step_d_loss = train_discriminator(X_mb, Z_batch, T_mb, embedder, generator, discriminator)

        if itt % 1000 == 0:
            print(f"Iter: {itt}, D_loss: {step_d_loss.numpy():.4f}")

    print('Finish Joint Training')
    print('hello')
    # Random Generator Function
    def random_generator(No, z_dim, dataT, Max_Seq_Len):
        Z_mb = []
        for i in range(No):
            Temp = np.zeros([Max_Seq_Len, z_dim], dtype=np.float32)
            Temp_Z = np.random.uniform(0., 1., [dataT[i], z_dim])
            Temp[:dataT[i], :] = Temp_Z
            Z_mb.append(Temp)
        return tf.convert_to_tensor(Z_mb, dtype=tf.float32)
    Z_mb = random_generator(No, z_dim, dataT, Max_Seq_Len)
    H_hat = generator(Z_mb)
    X_hat = recovery(H_hat)

    dataX_hat = list()
    
    for i in range(No):
        Temp = X_hat[i,:dataT[i],:]
        dataX_hat.append(Temp)
        
    # Renormalization
    if (Normalization_Flag == 1):
        dataX_hat = dataX_hat * max_val
        dataX_hat = dataX_hat + min_val
    
    
    return dataX_hat
