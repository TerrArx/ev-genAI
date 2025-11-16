import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import os
import sklearn  # <-- FIX 1: Required for joblib to load sklearn objects
import json     # <-- FIX 2: Required to load budget_bins.json

@st.cache_resource
def load_models_and_artifacts():
    """
    Loads all necessary models and preprocessing objects
    using absolute paths.
    """
    
    # --- THIS IS THE FIX ---
    # Get the absolute path of the directory where this script (cvae.py) is located
    # e.g., 'C:\Users\nabil\OneDrive\Desktop\EV-GenAI'
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Build absolute paths from this script's location
    DATA_PATH = os.path.join(SCRIPT_DIR, 'data-encoded-ev')
    MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'keras format')
    # --- End of Fix ---
    
    # --- Load Preprocessing Objects ---
    scaler_y = joblib.load(os.path.join(DATA_PATH, 'scaler_y.pkl'))
    encoder_c = joblib.load(os.path.join(DATA_PATH, 'encoder_c.pkl'))
    y_features = joblib.load(os.path.join(DATA_PATH, 'y_features.pkl'))
    # Load the correct filename from your preprocessing script
    c_feature_names = joblib.load(os.path.join(DATA_PATH, 'c_features_names.pkl'))
    
    # --- FIX 2 (continued): Load the budget bins ---
    with open(os.path.join(DATA_PATH, 'budget_bins.json'), 'r') as f:
        budget_bins = json.load(f)

    # --- Load Keras Models ---
    encoder = keras.models.load_model(os.path.join(MODEL_PATH, 'encoder.keras'))
    decoder = keras.models.load_model(os.path.join(MODEL_PATH, 'decoder.keras'))
    
    # --- Define Model Dimensions (from your training script) ---
    Y_DIM = len(y_features)
    C_DIM = len(c_feature_names)
    LATENT_DIM = 12 # From your training script

    print("--- Artifacts Loaded Successfully ---")
    
    # --- FIX 2 (continued): Return the budget_bins and LATENT_DIM ---
    return (encoder, decoder, scaler_y, encoder_c, y_features, 
            c_feature_names, budget_bins, LATENT_DIM)