#!/usr/bin/env python
# coding: utf-8

#7


import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau

op_dir = "/kaggle/input/encoded-ev/encoded-ev"
os.makedirs(op_dir, exist_ok=True)
assert os.path.exists(op_dir), "Run preprocessing notebook first!"


#8


data = np.load(os.path.join(op_dir, "training_data.npz"))
Y_train, C_train = data["Y_train"], data["C_train"]
Y_val, C_val = data["Y_val"], data["C_val"]

scaler_y = joblib.load(os.path.join(op_dir, "scaler_y.pkl"))
c_feature_names = joblib.load(os.path.join(op_dir, "c_features_names.pkl"))
Y_features = joblib.load(os.path.join(op_dir, "y_features.pkl"))

print("Y_train:", Y_train.shape)
print("C_train:", C_train.shape)
print("Targets:", Y_features)
print("Conditioning features:", len(c_feature_names))


#9


Y_DIM = Y_train.shape[1]        # 3 (Range, Price, Battery Capacity)
C_DIM = C_train.shape[1]        # depends on OHE
LATENT_DIM = 12                 # increased for better expressive capacity
BATCH_SIZE = 64
EPOCHS = 150


#10


y_in = layers.Input(shape=(Y_DIM,))
c_in = layers.Input(shape=(C_DIM,))

enc_input = layers.Concatenate()([y_in, c_in])
x = layers.Dense(128, activation="relu")(enc_input)
x = layers.Dense(64, activation="relu")(x)

z_mean = layers.Dense(LATENT_DIM)(x)
z_log_var = layers.Dense(LATENT_DIM)(x)

def sample_z(args):
    mean, log_var = args
    eps = tf.random.normal(shape=tf.shape(mean))
    return mean + tf.exp(0.5 * log_var) * eps

z = layers.Lambda(sample_z)([z_mean, z_log_var])

encoder = Model([y_in, c_in], [z_mean, z_log_var, z], name="encoder")


#11


z_in = layers.Input(shape=(LATENT_DIM,))
c_in_d = layers.Input(shape=(C_DIM,))

dec_input = layers.Concatenate()([z_in, c_in_d])
x = layers.Dense(128, activation="relu")(dec_input)
x = layers.Dense(64, activation="relu")(x)

y_out = layers.Dense(Y_DIM)(x)

decoder = Model([z_in, c_in_d], y_out, name="decoder")


#12


class CVAE(Model):
    def __init__(self, encoder, decoder, beta_init=0.05):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = tf.Variable(beta_init, trainable=False, dtype=tf.float32)

    def train_step(self, data):
        (y_in, c_in), y_true = data
    
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([y_in, c_in], training=True)
            y_pred = self.decoder([z, c_in], training=True)
    
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            total_loss = recon_loss + self.beta * kl_loss
    
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "beta": self.beta
        }


    def test_step(self, data):
        (y_in, c_in), y_true = data
    
        z_mean, z_log_var, z = self.encoder([y_in, c_in], training=False)
        y_pred = self.decoder([z, c_in], training=False)
    
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        total_loss = recon_loss + self.beta * kl_loss
    
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "beta": self.beta
        }

cvae = CVAE(encoder, decoder)


#13


class BetaAnnealingCallback(tf.keras.callbacks.Callback):
    def __init__(self, beta_start=0.05, beta_end=1.0, warmup_epochs=40):
        super().__init__()
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        t = min(epoch / self.warmup_epochs, 1)
        beta = self.beta_start + (self.beta_end - self.beta_start) * (1 / (1 + np.exp(-6*(t - 0.5))))
        self.model.beta.assign(beta)


#14


cvae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

history = cvae.fit(
    x=[Y_train, C_train],
    y=Y_train,
    validation_data=([Y_val, C_val], Y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[
        BetaAnnealingCallback(beta_start=0.05, beta_end=1.0, warmup_epochs=40),
        ReduceLROnPlateau(monitor='val_total_loss', factor=0.5, patience=7, min_lr=1e-5)
    ],
    verbose=2
)


#15


# Encode validation set
z_mean, z_log_var, z = cvae.encoder.predict([Y_val, C_val], batch_size=256)

# Decode (reconstruct Y)
Y_val_pred = cvae.decoder.predict([z, C_val], batch_size=256)

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = np.mean(np.abs(Y_val - Y_val_pred), axis=0)
rmse = np.sqrt(np.mean((Y_val - Y_val_pred)**2, axis=0))

print("Per-feature MAE (standardized):", mae)
print("Per-feature RMSE (standardized):", rmse)


#17


from joblib import load

scaler_y = load("/kaggle/input/encoded-ev/encoded-ev/scaler_y.pkl")

Y_val_orig = scaler_y.inverse_transform(Y_val)
Y_val_pred_orig = scaler_y.inverse_transform(Y_val_pred)

mae_orig = np.mean(np.abs(Y_val_orig - Y_val_pred_orig), axis=0)
rmse_orig = np.sqrt(np.mean((Y_val_orig - Y_val_pred_orig)**2, axis=0))

print("Per-feature MAE (original units):", mae_orig)
print("Per-feature RMSE (original units):", rmse_orig)


#19


encoder.save(os.path.join('/kaggle/working/3', "encoder.h5"))
decoder.save(os.path.join('/kaggle/working/3', "decoder.h5"))
print(" Models saved.")


#21


cvae.encoder.save("encoder.keras")
cvae.decoder.save("decoder.keras")

