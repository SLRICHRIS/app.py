#!/usr/bin/env python3
"""
Streamlit VOCAPRA Audio Event Explorer

Features:
- Upload a WAV file
- Run MFCC(+Δ,+Δ²) feature extraction (same pipeline as training)
- Classify with a tiny Conv1D model trained on VOCAPRA
- Visualize class probabilities
- Generate a Grad-CAM-like heatmap over the feature timeline

Expected artifacts (from your training pipeline) in:
    vocapra_project/best_model.h5
    vocapra_project/label_to_idx.json
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import streamlit as st
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# -------------------------------------------------------------------
# Configuration (must match your training script)
# -------------------------------------------------------------------

ARTIFACT_DIR = Path("vocapra_project")
MODEL_PATH = ARTIFACT_DIR / "best_model.h5"
LABEL_MAP_PATH = ARTIFACT_DIR / "label_to_idx.json"

SR = 16000          # sampling rate
N_MFCC = 13         # base MFCCs
WIN_LEN = 0.025     # 25 ms window
HOP_LEN = 0.010     # 10 ms hop
TARGET_FRAMES = 80  # temporal length used during training

FEAT_DIM = N_MFCC * 3  # MFCC + Δ + Δ²

# -------------------------------------------------------------------
# Low-level feature pipeline
# -------------------------------------------------------------------


def compute_mfcc_with_deltas(
    y: np.ndarray,
    sr: int = SR,
    n_mfcc: int = N_MFCC,
    win_len: float = WIN_LEN,
    hop_len: float = HOP_LEN,
) -> np.ndarray:
    """Compute MFCC + Δ + Δ², returning (T, 3*n_mfcc)."""
    n_fft = int(win_len * sr)
    hop_length = int(hop_len * sr)

    mf = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )  # (n_mfcc, T)
    mf = mf.T  # (T, n_mfcc)

    d1 = librosa.feature.delta(mf.T).T
    d2 = librosa.feature.delta(mf.T, order=2).T

    feats = np.concatenate([mf, d1, d2], axis=1).astype(np.float32)
    return feats  # (T, 3*n_mfcc)


def prepare_fixed_length(feats: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    """
    Pad or truncate feature sequence to (1, target_frames, FEAT_DIM),
    right-aligning the real frames (same as training script).
    """
    T, F = feats.shape
    x = np.zeros((1, target_frames, F), dtype=np.float32)
    if T >= target_frames:
        x[0] = feats[:target_frames]
    else:
        x[0, -T:, :] = feats
    return x


# -------------------------------------------------------------------
# Model definition (must match training architecture)
# -------------------------------------------------------------------


def build_tiny_crnn_no_rnn(
    input_shape: Tuple[int, int],
    n_classes: int,
    dropout: float = 0.3,
) -> tf.keras.Model:
    """Tiny ConvNet (Conv1D + GlobalAveragePooling) used for VOCAPRA."""
    inp = layers.Input(shape=input_shape, name="input")
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(48, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inp, out, name="vocapra_tiny_convnet")
    return model


# -------------------------------------------------------------------
# Cached loaders
# -------------------------------------------------------------------


@st.cache_resource(show_spinner=True)
def load_label_map() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load label_to_idx and build idx_to_label."""
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(
            f"Label map not found at {LABEL_MAP_PATH}. "
            "Make sure you copied label_to_idx.json from your training run."
        )
    with open(LABEL_MAP_PATH, "r") as f:
        label_to_idx = json.load(f)
    idx_to_label = {idx: lab for lab, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


@st.cache_resource(show_spinner=True)
def load_model() -> Tuple[tf.keras.Model, Dict[int, str]]:
    """Rebuild the model architecture and load trained weights."""
    label_to_idx, idx_to_label = load_label_map()
    n_classes = len(label_to_idx)

    model = build_tiny_crnn_no_rnn(
        input_shape=(TARGET_FRAMES, FEAT_DIM),
        n_classes=n_classes,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}. "
            "Copy best_model.h5 from your training pipeline into vocapra_project/."
        )

    model.load_weights(str(MODEL_PATH))
    return model, idx_to_label


# -------------------------------------------------------------------
# Grad-CAM-like visualization
# -------------------------------------------------------------------


def grad_cam_for_sample(
    model: tf.keras.Model,
    x: np.ndarray,
    class_index: Optional[int] = None,
) -> np.ndarray:
    """
    Compute a Grad-CAM-like 1D saliency over time for a single example.

    Args:
        model: Keras model.
        x: input of shape (1, T, F).
        class_index: optional target class index. If None, uses argmax.
    Returns:
        cam_resized: np.ndarray of shape (T,) in [0, 1].
    """
    # Find last Conv1D layer
    conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv1D):
            conv_layer = layer
            break
    if conv_layer is None:
        raise RuntimeError("No Conv1D layer found in model for Grad-CAM.")

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [conv_layer.output, model.output],
    )

    x_tf = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        conv_outs, preds = grad_model(x_tf)
        if class_index is None:
            class_index = tf.argmax(preds[0]).numpy().item()
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_outs)          # (1, T', C)
    weights = tf.reduce_mean(grads, axis=1)         # (1, C)
    cam = tf.reduce_sum(conv_outs * weights[:, tf.newaxis, :], axis=-1)  # (1, T')
    cam = tf.nn.relu(cam).numpy()[0]

    # Normalize and resize to input time steps
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
    T_input = x.shape[1]
    cam_resized = np.interp(
        np.linspace(0, len(cam) - 1, T_input),
        np.arange(len(cam)),
        cam,
    )
    return cam_resized


# -------------------------------------------------------------------
# Visualization helpers
# -------------------------------------------------------------------


def plot_mfcc_and_cam(
    feats: np.ndarray,
    cam: np.ndarray,
    title: str,
) -> plt.Figure:
    """
    Create a figure showing MFCC feature map with Grad-CAM overlay.
    feats: (T, F)
    cam:  (T,)
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(feats.T, origin="lower", aspect="auto")
    ax.imshow(
        np.tile(cam, (feats.shape[1], 1)),
        origin="lower",
        aspect="auto",
        cmap="jet",
        alpha=0.45,
    )
    ax.set_title(title)
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Feature bins (MFCC + Δ + Δ²)")
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------


def main() -> None:
    # ---- Page config ------------------------------------------------
    st.set_page_config(
        page_title="VOCAPRA Audio Intelligence Console",
        page_icon="🎧",
        layout="wide",
    )

    st.title("🎧 VOCAPRA Audio Intelligence Console")
    st.markdown(
        """
        **Elite Technical Pipeline**

        - Upload a cow vocalization (WAV @ 16 kHz)
        - We run *MFCC + Δ + Δ²* feature extraction (same as training)
        - A tiny ConvNet (TFLite-friendly) predicts the behavioural class
        - A Grad-CAM-style saliency map shows **which time regions matter**
        """
    )

    with st.sidebar:
        st.header("Model & Pipeline")
        st.caption("All parameters must match your training script.")
        st.write(f"Sampling rate: `{SR} Hz`")
        st.write(f"MFCC: `{N_MFCC}` + deltas")
        st.write(f"Target frames: `{TARGET_FRAMES}`")
        st.write("Model: 2×Conv1D + GlobalAvgPool + Dense")

        st.markdown("---")
        st.info(
            "Make sure `vocapra_project/best_model.h5` and "
            "`vocapra_project/label_to_idx.json` are present in the repo."
        )

    # Load model once (cached)
    try:
        model, idx_to_label = load_model()
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

    # ---- File uploader ----------------------------------------------
    st.subheader("1️⃣ Upload an audio file")
    uploaded = st.file_uploader(
        "Upload a WAV file (16kHz mono recommended)",
        type=["wav"],
        accept_multiple_files=False,
    )

    if uploaded is None:
        st.info("Waiting for a WAV file...")
        return

    # Keep raw bytes for playback
    raw_bytes = uploaded.read()

    # Decode audio with librosa
    try:
        y, sr = librosa.load(io.BytesIO(raw_bytes), sr=SR, mono=True)
    except Exception as e:
        st.error(f"Failed to decode WAV with librosa: {e}")
        return

    if len(y) == 0:
        st.error("Audio appears to be empty.")
        return

    st.success(f"Loaded audio: {len(y) / SR:.2f} seconds @ {SR} Hz")

    # Display audio player and basic info
    col_audio, col_meta = st.columns([2, 1], gap="large")
    with col_audio:
        st.audio(raw_bytes, format="audio/wav")
    with col_meta:
        st.metric("Duration (s)", f"{len(y) / SR:.2f}")
        st.metric("Samples", f"{len(y):,}")

    # ---- Feature extraction -----------------------------------------
    st.subheader("2️⃣ Feature extraction: MFCC + Δ + Δ²")

    feats = compute_mfcc_with_deltas(y, sr=SR)  # (T, F=3*N_MFCC)
    x = prepare_fixed_length(feats, TARGET_FRAMES)

    st.caption(
        f"Raw feature shape: `{feats.shape}` → network input: `{x.shape}` "
        "(1, time, feature_dim)"
    )

    # ---- Prediction --------------------------------------------------
    st.subheader("3️⃣ Behaviour prediction")

    probs = model.predict(x)[0]  # (n_classes,)
    class_idx = int(np.argmax(probs))
    class_name = idx_to_label.get(class_idx, f"class_{class_idx}")
    confidence = float(probs[class_idx])

    st.markdown(
        f"""
        ### 🎯 Top prediction: **{class_name}**  
        Confidence: **{confidence * 100:.2f}%**
        """
    )

    # Probability bar chart
    labels = [idx_to_label[i] for i in range(len(probs))]
    prob_dict = {labels[i]: float(probs[i]) for i in range(len(probs))}
    st.bar_chart(prob_dict)

    # ---- Grad-CAM ----------------------------------------------------
    st.subheader("4️⃣ Grad-CAM temporal saliency")

    try:
        cam = grad_cam_for_sample(model, x, class_index=class_idx)
    except Exception as e:
        st.error(f"Grad-CAM computation failed: {e}")
        return

    fig = plot_mfcc_and_cam(feats, cam, title=f"Grad-CAM for predicted class: {class_name}")
    st.pyplot(fig, clear_figure=True)

    # ---- Technical details ------------------------------------------
    with st.expander("ℹ️ Deep technical details"):
        st.write("**Model summary**")
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        summary_str = "\n".join(stringlist)
        st.code(summary_str, language="text")

        st.write("**Class index mapping (idx → label)**")
        st.json(idx_to_label)


if __name__ == "__main__":
    main()
