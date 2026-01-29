"""Streamlit UI for chair type classification.

Provides a user interface for entering chair parameters, scaling inputs
with the saved preprocessing scaler, and running inference with a trained
model to demonstrate the end-to-end pipeline.
"""

import os
import pickle

import numpy as np
import streamlit as st
from tensorflow import keras


LABEL_MAP = {
    0: "Simple Chair",
    1: "Chair with Backrest",
    2: "Bar Chair",
    3: "Stool",
}

SCALER_PATH = os.path.join("config", "preprocessing_params.pkl")
MODEL_PATH = os.path.join("models", "trained_model.h5")


def load_scaler(path: str):
    """Load the preprocessing scaler from disk."""
    with open(path, "rb") as file_handle:
        return pickle.load(file_handle)


def load_model(path: str) -> keras.Model:
    """Load the trained Keras model from disk."""
    return keras.models.load_model(path)


@st.cache_resource
def load_artifacts() -> tuple[keras.Model, object]:
    """Load model and scaler once at app startup."""
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Scaler not found. Please run preprocessing first.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Please train the model first.")

    print(f"Loading model: {MODEL_PATH}")
    scaler = load_scaler(SCALER_PATH)
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
    return model, scaler


def build_input_array(values) -> np.ndarray:
    """Build a 2D numpy array from user inputs."""
    return np.array([values], dtype=float)


def main() -> None:
    """Render the Streamlit UI and run inference on demand."""
    st.title("Chair Type Classifier (Etapa 5.4)")
    st.write("Introduceți parametrii geometrici ai scaunului și apăsați Predict.")

    seat_height = st.number_input("seat_height", min_value=0.4, max_value=0.8, value=0.55, step=0.01)
    seat_width = st.number_input("seat_width", min_value=0.35, max_value=0.6, value=0.45, step=0.01)
    seat_depth = st.number_input("seat_depth", min_value=0.35, max_value=0.6, value=0.45, step=0.01)
    leg_count = st.number_input("leg_count", min_value=3, max_value=5, value=4, step=1)
    leg_thickness = st.number_input("leg_thickness", min_value=0.03, max_value=0.08, value=0.05, step=0.01)
    has_backrest = st.number_input("has_backrest", min_value=0, max_value=1, value=1, step=1)
    backrest_height = st.number_input("backrest_height", min_value=0.0, max_value=0.5, value=0.25, step=0.01)
    style_variant = st.number_input("style_variant", min_value=0, max_value=2, value=0, step=1)

    try:
        model, scaler = load_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    if st.button("Predict"):

        features = build_input_array(
            [
                seat_height,
                seat_width,
                seat_depth,
                leg_count,
                leg_thickness,
                has_backrest,
                backrest_height,
                style_variant,
            ]
        )
        scaled_features = scaler.transform(features)
        probabilities = model.predict(scaled_features, verbose=0)[0]
        predicted_label = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))

        st.success(f"Predicted class: {LABEL_MAP[predicted_label]}")
        st.info(f"Confidence: {confidence:.2f}")
        st.write("Probabilities:")
        st.json({LABEL_MAP[idx]: float(prob) for idx, prob in enumerate(probabilities)})


if __name__ == "__main__":
    main()
