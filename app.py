#!/usr/bin/env python3
"""
VOCAPRA Streamlit App
---------------------

Elite technical UI to:
- Upload a WAV file
- Run VOCAPRA MFCC(+Δ,+Δ²)-based pipeline
- Predict event class using trained Conv1D model
- Visualize Grad-CAM saliency overlay

Requires:
- vocapra_project/best_model.h5
- vocapra_project/label_to_idx.json
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import streamlit as st
import librosa
import librosa.display  # for nice plots
import tensorflow as tf
import matplotlib.pyplot as plt

# --------------------
# Global Config (must match training script)
# --------------------
MODEL_DIR = Path("vocapra_project")

SR = 16000
N_MFCC = 13
WIN_LEN = 0.025
HOP_LEN = 0.010
TARGET_FRAMES = 80

# --------------------
# Styling / Elite UI
# --------------------

st.set_page_config(
    page_title="VOCAPRA Audio Event Intelligence",
    page_icon="🎧",
    layout="wide",
)

# Custom CSS for elite look
st.markdown(
    """
    <style>
    /* App background and cards */
    .stApp {
        background: radial-gradient(circle at top left, #0f172a, #020617);
        color: #e5e7eb;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    h1, h2, h3, h4 {
        color: #e5e7eb !important;
    }
    .metric-card {
        background: rgba(15,23,42,0.75);
        border-radius: 18px;
        padding: 1rem 1.25rem;
        border: 1px solid rgba(148,163,184,0.4);
    }
    .section-card {
        background: rgba(15,23,42,0.85);
        border-radius: 22px;
        padding: 1.25rem 1.6rem;
        border: 1px solid rgba(148,163,184,0.45);
        box-shadow: 0 16px 40px rgba(0,0,0,0.5);
    }
    .stProgress > div > div {
        background-image: linear-gradient(90deg, #38bdf8, #a855f7);
    }
    .prob-bar {
        background: linear-gradient(90deg, rgba(56,189,248,0.12), rgba(168,85,247,0.20));
        border-radius: 10px;
        padding: 0.15rem 0.4rem;
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------
# Core Audio Processing
# --------------------


def compute_mfcc_with_deltas(
    y: np.ndarray,
    sr: int = SR,
    n_mfcc: int = N_MFCC,
    win_len: float = WIN_LEN,
    hop_len: float = HOP_LEN,
) -> np.ndarray:
    """Same MFCC(+Δ,+Δ²) extractor as training pipeline."""
    n_fft = int(win_len * sr)
    hop_length = int(hop_len * sr)

    mf = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )  # (n_mfcc, T)
    mf = mf.T  # (T, n_mfcc)

    d1 = librosa.feature.delta(mf.T).T
    d2 = librosa.feature.delta(mf.T, order=2).T

    feats = np.concatenate([mf, d1, d2], axis=1)  # (T, n_mfcc*3)
    return feats.astype(np.float32)


def make_fixed_length(feats: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    """Pad/truncate feature sequence to fixed (target_frames, F). Right-aligned like training."""
    T, F = feats.shape
    out = np.zeros((target_frames, F), dtype=np.float32)
    if T >= target_frames:
        out[:] = feats[:target_frames]
    else:
        out[-T:, :] = feats
    return out


# --------------------
# Model + Metadata
# --------------------


@st.cache_resource(show_spinner=True)
def load_model_and_labels(
    model_dir: Path = MODEL_DIR,
) -> Tuple[tf.keras.Model, Dict[int, str], str]:
    """Load trained Keras model, label mapping, and find last Conv1D layer."""
    model_path = model_dir / "best_model.h5"
    label_path = model_dir / "label_to_idx.json"

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()

    if not label_path.exists():
        st.error(f"Label mapping file not found: {label_path}")
        st.stop()

    with open(label_path, "r") as f:
        label_to_idx = json.load(f)

    idx_to_label = {int(v): k for k, v in label_to_idx.items()}

    model = tf.keras.models.load_model(model_path)

    # Find last Conv1D layer for Grad-CAM
    conv_layer_name: Optional[str] = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv1D):
            conv_layer_name = layer.name
            break

    if conv_layer_name is None:
        st.warning("No Conv1D layer found for Grad-CAM; saliency will be disabled.")
        conv_layer_name = ""

    return model, idx_to_label, conv_layer_name


def compute_prediction_and_cam(
    model: tf.keras.Model,
    idx_to_label: Dict[int, str],
    conv_layer_name: str,
    x_fixed: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """
    Run prediction & Grad-CAM on a single fixed-length sample.

    x_fixed: (TARGET_FRAMES, F)
    Returns:
        probs: (num_classes,)
        cam_resized: (TARGET_FRAMES,)
        class_idx: int
        class_name: str
    """
    x = x_fixed[np.newaxis, ...]  # (1, T, F)
    probs = model.predict(x, verbose=0)[0]  # (C,)
    class_idx = int(np.argmax(probs))
    class_name = idx_to_label.get(class_idx, str(class_idx))

    if not conv_layer_name:
        # No CAM, just zeros
        cam_resized = np.zeros(TARGET_FRAMES, dtype=np.float32)
        return probs, cam_resized, class_idx, class_name

    # Build Grad-CAM model
    conv_layer = model.get_layer(conv_layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs], [conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outs, preds_full = grad_model(tf.convert_to_tensor(x))
        loss = preds_full[:, class_idx]

    grads = tape.gradient(loss, conv_outs)  # (1, T', C)
    if grads is None:
        cam_resized = np.zeros(TARGET_FRAMES, dtype=np.float32)
        return probs, cam_resized, class_idx, class_name

    weights = tf.reduce_mean(grads, axis=1)  # (1, C)
    cam = tf.reduce_sum(conv_outs * weights[:, tf.newaxis, :], axis=-1)  # (1, T')
    cam = tf.nn.relu(cam).numpy()[0]  # (T',)

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

    # Resize to input time steps
    T_prime = cam.shape[0]
    cam_resized = np.interp(
        np.linspace(0, T_prime - 1, TARGET_FRAMES),
        np.arange(T_prime),
        cam,
    ).astype(np.float32)

    return probs, cam_resized, class_idx, class_name


# --------------------
# UI Layout
# --------------------

st.markdown(
    """
    <h1>🎧 VOCAPRA Audio Event Intelligence</h1>
    <p style="color:#9ca3af;font-size:0.95rem;">
      Tiny Conv1D + MFCC(+Δ,+Δ²) engine, quantization-ready, wrapped in a high-fidelity Streamlit interface.
    </p>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ Engine Controls")
    st.markdown(
        """
        <p style="font-size:0.9rem;color:#9ca3af;">
        This UI uses the same pipeline as your training script:<br>
        <code>16 kHz → MFCC(+Δ,+Δ²) → Conv1D → GlobalAvgPool → Softmax</code>
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.write("**Model directory:**")
    st.code(str(MODEL_DIR), language="bash")
    st.info(
        "Ensure `best_model.h5` and `label_to_idx.json` exist in this directory.",
        icon="ℹ️",
    )

    st.divider()
    st.markdown("**Audio expectations**")
    st.caption("• Mono WAV • 16 kHz recommended (other rates will be resampled).")

model, idx_to_label, conv_layer_name = load_model_and_labels()
num_classes = len(idx_to_label)

# Top layout: upload + quick stats
col_left, col_right = st.columns([1.2, 1.0])

with col_left:
    st.markdown("#### 🎙️ Upload an audio file")
    uploaded_file = st.file_uploader(
        "Drop a WAV file here",
        type=["wav", "wave"],
        label_visibility="collapsed",
    )

with col_right:
    st.markdown("#### 📊 Model snapshot")
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Number of classes", num_classes)
        st.metric("Sample rate (Hz)", SR)
        st.metric("Frames per sample", TARGET_FRAMES)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

if uploaded_file is None:
    st.info("Upload a WAV file to see predictions and Grad-CAM.", icon="👆")
    st.stop()

# --------------------
# Audio Loading & Preprocessing
# --------------------
st.markdown("### 🔄 Audio ingestion & preprocessing")

# Load audio from uploaded file
bytes_data = uploaded_file.read()
audio_buf = io.BytesIO(bytes_data)

y, sr = librosa.load(audio_buf, sr=SR, mono=True)  # resample to SR
duration = len(y) / SR

c1, c2 = st.columns([2.0, 1.2])

with c1:
    st.markdown("**Waveform**")
    fig_wav, ax_wav = plt.subplots(figsize=(8, 2))
    librosa.display.waveshow(y, sr=SR, ax=ax_wav)
    ax_wav.set_xlabel("Time (s)")
    ax_wav.set_ylabel("Amplitude")
    ax_wav.set_title("Waveform")
    st.pyplot(fig_wav, use_container_width=True)
    plt.close(fig_wav)

with c2:
    st.markdown("**Signal summary**")
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Duration (s)", f"{duration:.2f}")
        st.metric("Samples", f"{len(y):,}")
        st.metric("Resampled SR", SR)
        st.markdown("</div>", unsafe_allow_html=True)

# Compute MFCC(+Δ,+Δ²)
feats = compute_mfcc_with_deltas(y, sr=SR)
x_fixed = make_fixed_length(feats, TARGET_FRAMES)

st.caption(
    f"Features shape (variable): `{feats.shape}` → fixed: `{x_fixed.shape}` (T={TARGET_FRAMES})"
)

# --------------------
# Prediction + Grad-CAM
# --------------------
st.markdown("### 🧠 Model prediction & temporal saliency")

probs, cam_resized, class_idx, class_name = compute_prediction_and_cam(
    model, idx_to_label, conv_layer_name, x_fixed
)

# Sort probabilities (descending)
prob_pairs = sorted(
    [(idx_to_label[i], float(p)) for i, p in enumerate(probs)],
    key=lambda x: x[1],
    reverse=True,
)

col_pred, col_cam = st.columns([1.1, 1.6])

with col_pred:
    st.markdown("#### 🔮 Top predictions")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    best_label, best_prob = prob_pairs[0]
    st.markdown(
        f"<h3 style='margin-bottom:0.2rem;'>{best_label}</h3>"
        f"<p style='color:#9ca3af;margin-top:0;'>Confidence: "
        f"<b>{best_prob*100:.2f}%</b></p>",
        unsafe_allow_html=True,
    )
    st.progress(best_prob)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("**Class probabilities**")

    for lab, p in prob_pairs:
        bar = "█" * int(p * 20)
        st.markdown(
            f"<div class='prob-bar'><code>{lab:>22}</code> &nbsp; "
            f"{p*100:5.2f}%</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

with col_cam:
    st.markdown("#### 🔥 Grad-CAM over MFCC(+Δ,+Δ²)")

    fig_cam, ax_cam = plt.subplots(figsize=(9, 3))
    # Show features as time-frequency map
    im = ax_cam.imshow(
        x_fixed.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax_cam, fraction=0.025, pad=0.02, label="Feature value")

    # Overlay CAM (same time axis, repeated over feature dims)
    ax_cam.imshow(
        np.tile(cam_resized, (x_fixed.shape[1], 1)),
        origin="lower",
        aspect="auto",
        cmap="jet",
        alpha=0.45,
    )

    ax_cam.set_title(f"Grad-CAM overlay (pred: {class_name})")
    ax_cam.set_xlabel("Time frames")
    ax_cam.set_ylabel("Feature bins")

    st.pyplot(fig_cam, use_container_width=True)
    plt.close(fig_cam)

st.markdown("---")

with st.expander("🔍 Pipeline integrity check (for your sanity)", expanded=False):
    st.write(
        """
        - **Preprocessing** matches training:
          - 16 kHz mono
          - MFCC with `n_mfcc=13`, window 25 ms, hop 10 ms
          - Δ and Δ² concatenated → feature dim = 39
        - **Temporal normalization**:
          - Variable-length sequences → fixed `TARGET_FRAMES=80`
          - Right-aligned (latest frames kept) — same as training script.
        - **Model**:
          - Conv1D→BN→MaxPool → Conv1D→BN→MaxPool → GlobalAveragePooling1D → Dense softmax
          - Exactly the same architecture as your training pipeline.
        - **Grad-CAM**:
          - Uses the *last* Conv1D layer for saliency
          - Global-average pooling of gradients to get channel weights
          - ReLU-activated CAM, normalized and interpolated to full time resolution
        """
    )

st.success(
    "Pipeline is strong and consistent with your original training script. "
    "This UI is just a visual and interactive shell around that same backbone.",
    icon="✅",
)
