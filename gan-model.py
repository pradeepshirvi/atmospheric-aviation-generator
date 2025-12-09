"""
Generative Adversarial Network (GAN) for Synthetic Atmospheric Data Generation
Advanced ML model for creating highly realistic synthetic datasets
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import json

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class AtmosphericGAN:
    """
    GAN for generating synthetic atmospheric data that mimics real radiosonde patterns
    """
    
    def __init__(self, input_dim=7, latent_dim=100, learning_rate=0.0002):
        """
        Initialize the GAN model
        
        Args:
            input_dim: Number of atmospheric variables (temp, pressure, humidity, etc.)
            latent_dim: Dimension of the noise vector
            learning_rate: Learning rate for optimizers
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        
        # Build models
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gan = self._build_gan()
        
        # Data scaler
        self.scaler = StandardScaler()
        
        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'd_acc': []
        }
    
    def _build_generator(self):
        """Build the generator network"""
        model = models.Sequential([
            layers.Dense(256, input_dim=self.latent_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(momentum=0.8),
            
            layers.Dense(512),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(momentum=0.8),
            
            layers.Dense(1024),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(momentum=0.8),
            
            layers.Dense(self.input_dim, activation='tanh')
        ])
        
        return model
    
    def _build_discriminator(self):
        """Build the discriminator network"""
        model = models.Sequential([
            layers.Dense(512, input_dim=self.input_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            layers.Dense(256),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            layers.Dense(128),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile discriminator
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5),
            metrics=['accuracy']
        )
        
        return model
    
    def _build_gan(self):
        """Build the combined GAN model"""
        # Freeze discriminator weights for generator training
        self.discriminator.trainable = False
        
        # Build GAN: Generator -> Discriminator
        model = models.Sequential([
            self.generator,
            self.discriminator
        ])
        
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5)
        )
        
        return model
    
    def train(self, X_train, epochs=1000, batch_size=32, verbose=1):
        """
        Train the GAN model
        
        Args:
            X_train: Training data (real atmospheric measurements)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
        """
        # Normalize the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Labels for real and fake data
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # ---------------------
            # Train Discriminator
            # ---------------------
            
            # Select random batch of real data
            idx = np.random.randint(0, X_train_scaled.shape[0], batch_size)
            real_data = X_train_scaled[idx]
            
            # Generate fake data
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_data = self.generator.predict(noise, verbose=0)
            
            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch(real_data, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_data, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            # Train Generator
            # ---------------------
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, real_labels)
            
            # Store history
            self.history['d_loss'].append(d_loss[0])
            self.history['d_acc'].append(d_loss[1])
            self.history['g_loss'].append(g_loss)
            
            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs} - D loss: {d_loss[0]:.4f}, "
                      f"D acc: {d_loss[1]:.4f}, G loss: {g_loss:.4f}")
    
    def generate_samples(self, n_samples):
        """
        Generate synthetic atmospheric data samples
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Generated samples (denormalized)
        """
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        generated_scaled = self.generator.predict(noise, verbose=0)
        generated = self.scaler.inverse_transform(generated_scaled)
        
        return generated
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.generator.save(f"{filepath}_generator.h5")
        self.discriminator.save(f"{filepath}_discriminator.h5")
        
        # Save scaler
        with open(f"{filepath}_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save history
        with open(f"{filepath}_history.json", 'w') as f:
            json.dump(self.history, f)
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.generator = models.load_model(f"{filepath}_generator.h5")
        self.discriminator = models.load_model(f"{filepath}_discriminator.h5")
        
        # Load scaler
        with open(f"{filepath}_scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load history
        with open(f"{filepath}_history.json", 'r') as f:
            self.history = json.load(f)

class TimeSeriesVAE:
    """
    Variational Autoencoder for generating time-series atmospheric data
    """
    
    def __init__(self, sequence_length=50, n_features=7, latent_dim=20):
        """
        Initialize VAE for time-series data
        
        Args:
            sequence_length: Length of time sequences
            n_features: Number of features per time step
            latent_dim: Dimension of latent space
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.latent_dim = latent_dim
        
        # Build model
        self.encoder, self.decoder, self.vae = self._build_vae()
        
        # Data scaler
        self.scaler = StandardScaler()
    
    def _sampling(self, args):
        """Reparameterization trick for VAE"""
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def _build_vae(self):
        """Build the VAE architecture"""
        # Encoder
        encoder_inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM layers for temporal processing
        x = layers.LSTM(128, return_sequences=True)(encoder_inputs)
        x = layers.LSTM(64)(x)
        
        # Latent space
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        # Sampling layer
        z = layers.Lambda(self._sampling, output_shape=(self.latent_dim,), 
                         name='z')([z_mean, z_log_var])
        
        encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        
        # Decoder
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(64)(latent_inputs)
        x = layers.RepeatVector(self.sequence_length)(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LSTM(128, return_sequences=True)(x)
        decoder_outputs = layers.TimeDistributed(layers.Dense(self.n_features))(x)
        
        decoder = models.Model(latent_inputs, decoder_outputs, name='decoder')
        
        # VAE model
        outputs = decoder(encoder(encoder_inputs)[2])
        vae = models.Model(encoder_inputs, outputs, name='vae')
        
        # Custom VAE loss
        reconstruction_loss = tf.keras.losses.mse(encoder_inputs, outputs)
        reconstruction_loss *= self.sequence_length * self.n_features
        
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss) * -0.5
        
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        
        vae.compile(optimizer='adam')
        
        return encoder, decoder, vae
    
    def train(self, X_train, epochs=100, batch_size=32):
        """Train the VAE model"""
        # Reshape data for time series
        n_samples = len(X_train) // self.sequence_length
        X_train_reshaped = X_train[:n_samples * self.sequence_length].reshape(
            n_samples, self.sequence_length, self.n_features
        )
        
        # Normalize
        X_train_flat = X_train_reshaped.reshape(-1, self.n_features)
        X_train_flat_scaled = self.scaler.fit_transform(X_train_flat)
        X_train_scaled = X_train_flat_scaled.reshape(
            n_samples, self.sequence_length, self.n_features
        )
        
        # Train
        history = self.vae.fit(
            X_train_scaled, X_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    
    def generate_sequences(self, n_sequences):
        """Generate new time series sequences"""
        # Sample from latent space
        latent_samples = np.random.normal(0, 1, (n_sequences, self.latent_dim))
        
        # Generate sequences
        generated_scaled = self.decoder.predict(latent_samples)
        
        # Denormalize
        generated_flat = generated_scaled.reshape(-1, self.n_features)
        generated = self.scaler.inverse_transform(generated_flat)
        generated_sequences = generated.reshape(n_sequences, self.sequence_length, self.n_features)
        
        return generated_sequences

class HybridSyntheticGenerator:
    """
    Hybrid approach combining physics-based models with ML
    """
    
    def __init__(self):
        self.gan = AtmosphericGAN()
        self.vae = TimeSeriesVAE()
        self.physics_constraints = {}
    
    def apply_physics_constraints(self, data):
        """
        Apply physical constraints to ensure realistic data
        
        Args:
            data: Generated data array
            
        Returns:
            Physically constrained data
        """
        df = pd.DataFrame(data, columns=['altitude', 'temperature', 'pressure', 
                                        'humidity', 'wind_speed', 'wind_dir', 'dewpoint'])
        
        # Temperature-altitude constraint (lapse rate)
        for i in range(1, len(df)):
            alt_diff = df.loc[i, 'altitude'] - df.loc[i-1, 'altitude']
            if alt_diff > 0:
                # Temperature should decrease with altitude
                expected_temp_change = -0.0065 * alt_diff  # Standard lapse rate
                df.loc[i, 'temperature'] = df.loc[i-1, 'temperature'] + expected_temp_change + np.random.normal(0, 0.5)
        
        # Pressure-altitude constraint (barometric formula)
        for i in range(len(df)):
            alt = df.loc[i, 'altitude']
            # Approximate pressure using barometric formula
            df.loc[i, 'pressure'] = 1013.25 * (1 - 0.0065 * alt / 288.15) ** 5.255
            df.loc[i, 'pressure'] += np.random.normal(0, 2)  # Add small variation
        
        # Humidity constraints
        df['humidity'] = df['humidity'].clip(0, 100)
        
        # Wind speed constraints
        df['wind_speed'] = df['wind_speed'].clip(0, 200)
        
        # Wind direction constraints
        df['wind_dir'] = df['wind_dir'] % 360
        
        # Dewpoint must be <= temperature
        df['dewpoint'] = np.minimum(df['dewpoint'], df['temperature'] - 1)
        
        return df.values
    
    def generate_hybrid_dataset(self, n_samples, use_physics=True):
        """
        Generate dataset using hybrid approach
        
        Args:
            n_samples: Number of samples to generate
            use_physics: Whether to apply physics constraints
            
        Returns:
            Generated dataset
        """
        # Generate using GAN
        gan_samples = self.gan.generate_samples(n_samples)
        
        if use_physics:
            # Apply physics constraints
            constrained_samples = self.apply_physics_constraints(gan_samples)
            return constrained_samples
        else:
            return gan_samples

# Example usage and testing
if __name__ == "__main__":
    print("Initializing Advanced Synthetic Data Generator with GANs...")
    
    # Generate sample training data (in practice, use real radiosonde data)
    print("\nGenerating sample training data...")
    n_samples = 5000
    training_data = np.random.randn(n_samples, 7)
    
    # Add some structure to make it more realistic
    training_data[:, 0] = np.linspace(0, 20000, n_samples)  # Altitude
    training_data[:, 1] = 15 - 0.0065 * training_data[:, 0] + np.random.normal(0, 2, n_samples)  # Temperature
    training_data[:, 2] = 1013.25 * (1 - 0.0065 * training_data[:, 0] / 288.15) ** 5.255  # Pressure
    training_data[:, 3] = 70 * np.exp(-training_data[:, 0] / 8000) + np.random.normal(0, 5, n_samples)  # Humidity
    training_data[:, 4] = 5 + training_data[:, 0] / 1000 + np.random.normal(0, 3, n_samples)  # Wind speed
    training_data[:, 5] = 270 + np.random.normal(0, 30, n_samples)  # Wind direction
    training_data[:, 6] = training_data[:, 1] - 5 + np.random.normal(0, 2, n_samples)  # Dewpoint
    
    # Initialize and train GAN
    print("\nInitializing GAN model...")
    gan = AtmosphericGAN(input_dim=7)
    
    print("Training GAN (this may take a few minutes)...")
    gan.train(training_data, epochs=500, batch_size=32, verbose=1)
    
    # Generate synthetic samples
    print("\nGenerating synthetic samples...")
    synthetic_data = gan.generate_samples(100)
    
    # Create DataFrame for display
    columns = ['Altitude (m)', 'Temperature (°C)', 'Pressure (hPa)', 
               'Humidity (%)', 'Wind Speed (m/s)', 'Wind Direction (°)', 'Dewpoint (°C)']
    synthetic_df = pd.DataFrame(synthetic_data, columns=columns)
    
    print("\nFirst 10 synthetic samples:")
    print(synthetic_df.head(10))
    
    print("\nStatistical comparison:")
    print("\nOriginal data statistics:")
    original_df = pd.DataFrame(training_data, columns=columns)
    print(original_df.describe())
    
    print("\nSynthetic data statistics:")
    print(synthetic_df.describe())
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(gan.history['d_loss'])
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(gan.history['g_loss'])
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(gan.history['d_acc'])
    plt.title('Discriminator Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('gan_training_history.png')
    print("\nTraining history plot saved as 'gan_training_history.png'")
    
    # Test hybrid generator
    print("\n" + "="*50)
    print("Testing Hybrid Generator with Physics Constraints...")
    hybrid_gen = HybridSyntheticGenerator()
    hybrid_gen.gan = gan
    
    hybrid_data = hybrid_gen.generate_hybrid_dataset(100, use_physics=True)
    hybrid_df = pd.DataFrame(hybrid_data, columns=columns)
    
    print("\nHybrid generated data (with physics constraints):")
    print(hybrid_df.head(10))
    
    # Save model
    print("\nSaving trained model...")
    gan.save_model('atmospheric_gan_model')
    print("Model saved successfully!")
