# Implementation of tgan.py with up to date libraries

import numpy as np
import tensorflow as tf
import pickle
import time
from models import Embedder, Recovery, Generator, Supervisor, Discriminator


physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("TensorFlow is using Metal for GPU acceleration.")
    for device in physical_devices:
        print(f"GPU device detected: {device}")
else:
    print("No GPU found. TensorFlow is using CPU.")

# Min Max Normalizer
def MinMaxScaler(dataX):
    
    min_val = np.min(np.min(dataX, axis = 0), axis = 0)
    dataX = dataX - min_val
    
    max_val = np.max(np.max(dataX, axis = 0), axis = 0)
    dataX = dataX / (max_val + 1e-7)
    
    return dataX, min_val, max_val

def tgan(dataX, parameters, it):
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
    start = time.time()
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
            print(f"Iter: {itt} took {time.time() - start}s, E_loss: {step_e_loss.numpy():.4f}")
            start = time.time()
    
    # Just a lambda function to generate random noise for batch_size
    Z_mb = lambda batch_size: tf.random.uniform((batch_size, Max_Seq_Len, z_dim))

    # Supervised Training
    start = time.time()
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
            print(f"Iter: {itt} took {time.time() - start}s, G_loss: {step_g_loss.numpy():.4f}")
            start = time.time()

    # Joint Training
    start = time.time()
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
            print(f"Iter: {itt} took {time.time() - start}s, D_loss: {step_d_loss.numpy():.4f}")
            start = time.time()

    print('Finish Joint Training')

    # # Random Generator Function
    # def random_generator(No, z_dim, dataT, Max_Seq_Len):
    #     Z_mb = []
    #     for i in range(No):
    #         Temp = np.zeros([Max_Seq_Len, z_dim], dtype=np.float32)
    #         Temp_Z = np.random.uniform(0., 1., [dataT[i], z_dim])
    #         Temp[:dataT[i], :] = Temp_Z
    #         Z_mb.append(Temp)
    #     return tf.convert_to_tensor(Z_mb, dtype=tf.float32)
    # Z_mb = random_generator(No, z_dim, dataT, Max_Seq_Len)
    # H_hat = generator(Z_mb)
    # X_hat = recovery(H_hat)

    # dataX_hat = list()
    
    # for i in range(No):
    #     Temp = X_hat[i,:dataT[i],:]
    #     dataX_hat.append(Temp)
        
    # # Renormalization
    # if (Normalization_Flag == 1):
    #     dataX_hat = dataX_hat * max_val
    #     dataX_hat = dataX_hat + min_val

    # return dataX_hat

    generator.save(f"models/generator_{it}.keras")
    discriminator.save(f"models/discriminator_{it}.keras")
    recovery.save(f"models/recovery_{it}.keras")
    supervisor.save(f"models/supervisor_{it}.keras")
    embedder.save(f"models/embedder_{it}.keras")

        # Random Generator Function for Forecasting
    def generate_forecast(generator, recovery, steps=50, z_dim=10, max_val=1.0, min_val=0.0, normalization_flag=1):
        """
        Generates a forecast for a fixed number of future time steps.

        Parameters:
        - generator: The generator model.
        - recovery: The recovery model (maps latent space to data space).
        - steps: Number of time steps to forecast.
        - z_dim: Dimensionality of the latent space.
        - max_val, min_val: Values for renormalization.
        - normalization_flag: Whether to apply renormalization.

        Returns:
        - forecast: The generated forecast for the given number of steps.
        """
        # Generate random noise for forecasting
        noise = np.random.uniform(0., 1., (steps, z_dim)).astype(np.float32)
        noise_tensor = tf.convert_to_tensor([noise], dtype=tf.float32)  # Add batch dimension

        # Pass through generator and recovery models
        H_hat = generator(noise_tensor)  # Latent space output
        X_hat = recovery(H_hat)         # Map latent space back to data space

        # Extract forecast (remove batch dimension)
        forecast = X_hat.numpy()[0]  # Shape: (steps, feature_dim)

        # Renormalization
        if normalization_flag == 1:
            pkl_file_path = "normalization_params.pkl"
            with open(pkl_file_path, 'rb') as file:
                normalization_params = pickle.load(file)
                recovered_max_val = normalization_params["max_val"]
                recovered_min_val = normalization_params["min_val"]
            forecast = forecast * (recovered_max_val - recovered_min_val) + recovered_min_val

        return forecast

    forecast = generate_forecast(generator, recovery, steps=24, z_dim=z_dim, max_val=1.0, min_val=0.0, normalization_flag=0)
    return forecast
    
    
