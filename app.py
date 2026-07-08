"""TumorTrace — Streamlit app for interactive brain tumor segmentation.

Visual language is a "radiology workstation" console, not a generic SaaS
dashboard: flat bordered panels, monospace technical labels, a dark
teal-on-charcoal palette, no gradients or rounded glass cards.
"""
import os
import shutil
import tempfile
from glob import glob

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import streamlit as st

from constants import BEST_CHECKPOINT_PATH, DISCLAIMER, MODALITIES, OVERLAY_ALPHA, OVERLAY_COLORS, SAMPLES_DIR
from inference import compute_region_volumes_cm3, load_model, predict_full_volume
from preprocess import load_patient_volumes

st.set_page_config(page_title="TumorTrace", page_icon="🧠", layout="wide")


# ---------------------------------------------------------------------------
# Cached, expensive resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    return load_model(BEST_CHECKPOINT_PATH, device="cpu" if not _has_cuda() else None)


def _has_cuda():
    import torch
    return torch.cuda.is_available()


@st.cache_data
def list_sample_cases():
    paths = sorted(glob(os.path.join(SAMPLES_DIR, "*.npz")))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in paths}


def load_sample_volume(npz_path):
    data = np.load(npz_path)
    return data["image"].astype(np.float32), tuple(data["zooms"]), data["affine"].astype(np.float64)


def load_uploaded_volume(uploaded_files):
    """uploaded_files: dict {modality: UploadedFile}. Saves to a temp patient
    directory (named to match the modality-suffix convention preprocess.py
    expects) and loads it through the shared preprocess.load_patient_volumes
    path, so uploads go through the exact same normalization as training."""
    tmp_dir = tempfile.mkdtemp(prefix="tumortrace_upload_")
    try:
        for modality, uploaded_file in uploaded_files.items():
            out_path = os.path.join(tmp_dir, f"upload_{modality}.nii.gz")
            with open(out_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        image, _, zooms, affine = load_patient_volumes(tmp_dir, load_seg=False)
        return image, zooms, affine
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_inference(image, cache_key):
    if st.session_state.get("volume_key") != cache_key:
        model, device = get_model()
        with st.spinner("Running segmentation model on all slices (CPU, one-time per volume)..."):
            pred_mask = predict_full_volume(image, model, device)
        st.session_state["volume_key"] = cache_key
        st.session_state["pred_mask"] = pred_mask
        st.session_state["image"] = image
    return st.session_state["pred_mask"]


def default_slice_index(pred_mask):
    tumor_per_slice = (pred_mask > 0).sum(axis=(0, 1))
    if tumor_per_slice.max() == 0:
        return pred_mask.shape[-1] // 2
    return int(np.argmax(tumor_per_slice))


def overlay_rgb(base_gray, label_slice):
    rgb = np.stack([base_gray] * 3, axis=-1)
    for label, color in OVERLAY_COLORS.items():
        mask = label_slice == label
        for c in range(3):
            rgb[..., c] = np.where(mask, (1 - OVERLAY_ALPHA) * rgb[..., c] + OVERLAY_ALPHA * color[c], rgb[..., c])
    return np.clip(rgb, 0, 1)


def normalize_for_display(slice_2d):
    lo, hi = np.percentile(slice_2d, 1), np.percentile(slice_2d, 99)
    if hi <= lo:
        return np.zeros_like(slice_2d)
    return np.clip((slice_2d - lo) / (hi - lo), 0, 1)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧠 TumorTrace")
st.markdown(
    "**TumorTrace draws the exact boundary of a brain tumor on an MRI scan in seconds — "
    "trained on 369 real glioma patients from the BraTS 2020 challenge.**"
)
st.warning(DISCLAIMER)

# ---------------------------------------------------------------------------
# Input mode
# ---------------------------------------------------------------------------
mode = st.radio("Input", ["Try a sample case", "Upload your own"], horizontal=True)

image = None
cache_key = None

if mode == "Try a sample case":
    samples = list_sample_cases()
    if not samples:
        st.error(
            f"No bundled sample cases found in `{SAMPLES_DIR}/`. "
            "Run `python make_samples.py` (see README) to generate them."
        )
    else:
        case_name = st.selectbox("Choose a sample case", list(samples.keys()))
        image, zooms, affine = load_sample_volume(samples[case_name])
        cache_key = f"sample:{case_name}"

else:
    st.caption("Upload 4 co-registered NIfTI volumes (.nii.gz) for the same patient.")
    col1, col2, col3, col4 = st.columns(4)
    uploaders = {}
    for col, modality in zip([col1, col2, col3, col4], MODALITIES):
        with col:
            uploaders[modality] = st.file_uploader(modality.upper(), type=["nii", "gz"], key=f"upload_{modality}")

    if all(uploaders.values()):
        with st.spinner("Loading and normalizing uploaded volumes..."):
            image, zooms, affine = load_uploaded_volume(uploaders)
        cache_key = "upload:" + ":".join(f.name for f in uploaders.values())
    else:
        st.info("Upload all 4 modalities (T1, T1ce, T2, FLAIR) to run segmentation.")

# ---------------------------------------------------------------------------
# Inference + visualization
# ---------------------------------------------------------------------------
if image is not None:
    pred_mask = run_inference(image, cache_key)

    if st.session_state.get("last_key_for_slider") != st.session_state.get("volume_key"):
        st.session_state["slice_idx"] = default_slice_index(pred_mask)
        st.session_state["last_key_for_slider"] = st.session_state.get("volume_key")

    num_slices = pred_mask.shape[-1]
    slice_idx = st.slider("Axial slice", 0, num_slices - 1, key="slice_idx")

    flair_idx = MODALITIES.index("flair")
    flair_slice = normalize_for_display(image[flair_idx, :, :, slice_idx])
    pred_slice = pred_mask[:, :, slice_idx]

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("FLAIR (raw)")
        fig, ax = plt.subplots()
        ax.imshow(flair_slice, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.subheader("Predicted tumor overlay")
        fig, ax = plt.subplots()
        ax.imshow(overlay_rgb(flair_slice, pred_slice))
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    st.caption("🔴 enhancing tumor · 🟡 edema · 🔵 necrotic / non-enhancing core")

    st.subheader("Tumor volume summary")
    volumes = compute_region_volumes_cm3(pred_mask, zooms)
    m1, m2, m3 = st.columns(3)
    m1.metric("Whole Tumor (WT)", f"{volumes['WT']:.1f} cm³")
    m2.metric("Tumor Core (TC)", f"{volumes['TC']:.1f} cm³")
    m3.metric("Enhancing Tumor (ET)", f"{volumes['ET']:.1f} cm³")

    tmp_path = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    nib.save(nib.Nifti1Image(pred_mask.astype(np.uint8), affine), tmp_path)
    with open(tmp_path, "rb") as f:
        st.download_button(
            "Download predicted mask (.nii.gz)",
            data=f.read(),
            file_name="tumortrace_prediction.nii.gz",
            mime="application/gzip",
        )
    os.remove(tmp_path)

st.markdown("---")
st.caption(
    "TumorTrace is a research/educational prototype (see disclaimer above). "
    "Model: ResNet34-encoder U-Net trained on BraTS 2020. Not for clinical use."
)
