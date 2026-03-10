"""
Model Inference Module
Loads pre-trained GAN/VAE models and generates synthetic atmospheric data.
Falls back to physics-based generation if models are not available.
"""

import numpy as np
import pandas as pd
import os
import pickle
import json
from typing import Dict, Optional, Tuple

# Try importing TensorFlow (may not be available in all environments)
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Try importing sklearn
try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Path to saved models
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'trained_models')


class ModelManager:
    """
    Manages loading and inference for GAN and VAE models.
    Provides graceful fallback to physics-based generation.
    """

    def __init__(self):
        self.gan_generator = None
        self.gan_scaler = None
        self.vae_decoder = None
        self.vae_scaler = None
        self.gan_loaded = False
        self.vae_loaded = False
        self.gan_history = None
        self.vae_config = None

        # Column definitions
        self.gan_columns = [
            'altitude_m', 'temperature_c', 'pressure_hpa',
            'humidity_percent', 'wind_speed_mps', 'wind_direction_deg', 'dewpoint_c'
        ]
        self.vae_columns = [
            'altitude_m', 'temperature_c', 'pressure_hpa',
            'humidity_percent', 'wind_speed_mps', 'wind_direction_deg', 'dewpoint_c'
        ]

        # Try to load models on initialization
        self._load_models()

    def _load_models(self):
        """Attempt to load pre-trained models from disk"""
        if not TF_AVAILABLE:
            print("[ModelManager] TensorFlow not available - ML models disabled")
            return

        if not os.path.exists(MODELS_DIR):
            print(f"[ModelManager] Models directory not found: {MODELS_DIR}")
            return

        # Load GAN
        gan_gen_path = os.path.join(MODELS_DIR, 'atmospheric_gan_generator.h5')
        gan_scaler_path = os.path.join(MODELS_DIR, 'atmospheric_gan_scaler.pkl')
        gan_history_path = os.path.join(MODELS_DIR, 'atmospheric_gan_history.json')

        if os.path.exists(gan_gen_path) and os.path.exists(gan_scaler_path):
            try:
                self.gan_generator = keras.models.load_model(gan_gen_path, compile=False)
                with open(gan_scaler_path, 'rb') as f:
                    self.gan_scaler = pickle.load(f)
                if os.path.exists(gan_history_path):
                    with open(gan_history_path, 'r') as f:
                        self.gan_history = json.load(f)
                self.gan_loaded = True
                print("[ModelManager] GAN model loaded successfully")
            except Exception as e:
                print(f"[ModelManager] Failed to load GAN: {e}")

        # Load VAE
        vae_decoder_path = os.path.join(MODELS_DIR, 'atmospheric_vae_decoder.h5')
        vae_scaler_path = os.path.join(MODELS_DIR, 'atmospheric_vae_scaler.pkl')
        vae_config_path = os.path.join(MODELS_DIR, 'atmospheric_vae_config.json')

        if os.path.exists(vae_decoder_path) and os.path.exists(vae_scaler_path):
            try:
                self.vae_decoder = keras.models.load_model(vae_decoder_path, compile=False)
                with open(vae_scaler_path, 'rb') as f:
                    self.vae_scaler = pickle.load(f)
                if os.path.exists(vae_config_path):
                    with open(vae_config_path, 'r') as f:
                        self.vae_config = json.load(f)
                self.vae_loaded = True
                print("[ModelManager] VAE model loaded successfully")
            except Exception as e:
                print(f"[ModelManager] Failed to load VAE: {e}")

    def get_status(self) -> Dict:
        """Return current model availability status"""
        return {
            'tensorflow_available': TF_AVAILABLE,
            'sklearn_available': SKLEARN_AVAILABLE,
            'gan_loaded': self.gan_loaded,
            'vae_loaded': self.vae_loaded,
            'models_dir': MODELS_DIR,
            'models_dir_exists': os.path.exists(MODELS_DIR),
        }

    def generate_gan(self, n_samples: int = 100, latent_dim: int = 100,
                     apply_physics: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Generate data using the pre-trained GAN model.

        Args:
            n_samples: Number of data points to generate
            latent_dim: Latent space dimension (must match trained model)
            apply_physics: Whether to apply physics constraints

        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        if not self.gan_loaded:
            raise RuntimeError("GAN model not loaded. Train the model first using the Colab notebook.")

        # Generate from latent space
        noise = np.random.normal(0, 1, (n_samples, latent_dim))
        generated_scaled = self.gan_generator.predict(noise, verbose=0)

        # Inverse transform to original scale
        generated = self.gan_scaler.inverse_transform(generated_scaled)

        # Create DataFrame
        df = pd.DataFrame(generated, columns=self.gan_columns)

        if apply_physics:
            df = self._apply_physics_constraints(df)

        # Round values
        for col in df.columns:
            df[col] = df[col].round(2)

        metadata = {
            'model': 'GAN',
            'n_samples': n_samples,
            'latent_dim': latent_dim,
            'physics_constrained': apply_physics,
        }

        return df, metadata

    def generate_vae(self, n_sequences: int = 5, apply_physics: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Generate time-series data using the pre-trained VAE model.

        Args:
            n_sequences: Number of sequences to generate
            apply_physics: Whether to apply physics constraints

        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        if not self.vae_loaded:
            raise RuntimeError("VAE model not loaded. Train the model first using the Colab notebook.")

        config = self.vae_config or {'latent_dim': 20, 'sequence_length': 50, 'n_features': 7}
        latent_dim = config.get('latent_dim', 20)
        sequence_length = config.get('sequence_length', 50)

        # Sample from latent space
        latent_samples = np.random.normal(0, 1, (n_sequences, latent_dim))

        # Generate sequences
        generated_scaled = self.vae_decoder.predict(latent_samples, verbose=0)

        # Flatten and inverse transform
        n_features = generated_scaled.shape[-1]
        generated_flat = generated_scaled.reshape(-1, n_features)
        generated = self.vae_scaler.inverse_transform(generated_flat)
        generated_sequences = generated.reshape(n_sequences, sequence_length, n_features)

        # Convert to DataFrame (flatten all sequences)
        all_data = []
        for seq_idx in range(n_sequences):
            for step_idx in range(sequence_length):
                row = {
                    'sequence_id': seq_idx,
                    'step': step_idx,
                }
                for feat_idx, col in enumerate(self.vae_columns):
                    row[col] = generated_sequences[seq_idx, step_idx, feat_idx]
                all_data.append(row)

        df = pd.DataFrame(all_data)

        if apply_physics:
            df = self._apply_physics_constraints(df)

        # Round values
        for col in self.vae_columns:
            if col in df.columns:
                df[col] = df[col].round(2)

        metadata = {
            'model': 'VAE',
            'n_sequences': n_sequences,
            'sequence_length': sequence_length,
            'total_records': len(df),
            'physics_constrained': apply_physics,
        }

        return df, metadata

    def _apply_physics_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply physical constraints to ensure realistic data"""
        df = df.copy()

        # Sort by altitude if present
        if 'altitude_m' in df.columns:
            df['altitude_m'] = df['altitude_m'].clip(0, 35000)

        # Temperature: lapse rate constraint
        if 'temperature_c' in df.columns and 'altitude_m' in df.columns:
            for i in range(len(df)):
                alt = df.loc[i, 'altitude_m'] if 'altitude_m' in df.columns else 0
                expected_temp = 15 - 6.5 * alt / 1000
                # Allow ±10°C deviation from expected
                df.loc[i, 'temperature_c'] = np.clip(
                    df.loc[i, 'temperature_c'],
                    expected_temp - 10,
                    expected_temp + 10
                )

        # Pressure: barometric constraint
        if 'pressure_hpa' in df.columns and 'altitude_m' in df.columns:
            for i in range(len(df)):
                alt = df.loc[i, 'altitude_m']
                expected_pressure = 1013.25 * (1 - 0.0065 * alt / 288.15) ** 5.255
                # Allow ±15% deviation
                df.loc[i, 'pressure_hpa'] = np.clip(
                    df.loc[i, 'pressure_hpa'],
                    expected_pressure * 0.85,
                    expected_pressure * 1.15
                )

        # Humidity: 0-100%
        if 'humidity_percent' in df.columns:
            df['humidity_percent'] = df['humidity_percent'].clip(5, 100)

        # Wind speed: non-negative, capped at 200 m/s
        if 'wind_speed_mps' in df.columns:
            df['wind_speed_mps'] = df['wind_speed_mps'].clip(0, 200)

        # Wind direction: 0-360
        if 'wind_direction_deg' in df.columns:
            df['wind_direction_deg'] = df['wind_direction_deg'] % 360

        # Dewpoint <= temperature
        if 'dewpoint_c' in df.columns and 'temperature_c' in df.columns:
            df['dewpoint_c'] = np.minimum(df['dewpoint_c'], df['temperature_c'] - 1)

        return df


# Singleton instance
model_manager = ModelManager()
