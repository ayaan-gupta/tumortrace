"""TumorTrace — Streamlit app for interactive brain tumor segmentation.

Visual language is a "radiology workstation" console, not a generic SaaS
dashboard: flat bordered panels, monospace technical labels, a dark
coral-on-charcoal palette (matching site/index.html), no gradients or
rounded glass cards.
"""
import os
import shutil
import tempfile
from glob import glob

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: the default "macosx" backend requires
                        # the main thread and hangs silently under Streamlit's worker thread
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import streamlit as st

from constants import (
    BEST_CHECKPOINT_PATH,
    DISCLAIMER,
    MODALITIES,
    OVERLAY_ALPHA,
    OVERLAY_COLORS,
    SAMPLES_DIR,
)
from inference import compute_region_volumes_cm3, load_model, predict_full_volume
from preprocess import load_patient_volumes
from viewer3d import build_viewer_html, encode_volumes

st.set_page_config(page_title="TumorTrace", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------------------
# Visual identity — flat, bordered, monospace-accented "workstation" theme.
# ---------------------------------------------------------------------------
def _hex(color_float_rgb):
    r, g, b = (int(round(c * 255)) for c in color_float_rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


REGION_HEX = {
    "necrotic": _hex(OVERLAY_COLORS[1]),
    "edema": _hex(OVERLAY_COLORS[2]),
    "enhancing": _hex(OVERLAY_COLORS[3]),
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {{
    --bg: #0b0f14;
    --panel: #12181f;
    --panel-border: #263241;
    --text: #dbe4ec;
    --text-dim: #7d8b99;
    --accent: #ff5a45;
    --warn: #e2a53e;
    --core: {REGION_HEX['necrotic']};
    --edema: {REGION_HEX['edema']};
    --enhancing: {REGION_HEX['enhancing']};
}}

html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
.stApp {{ background: var(--bg); }}
h1, h2, h3 {{ font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.02em; }}

/* flat bordered panels (st.container(border=True)) */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 3px !important;
    border: 1px solid var(--panel-border) !important;
    background: var(--panel) !important;
}}

/* header */
.tt-header {{ border-bottom: 1px solid var(--panel-border); padding-bottom: 0.9rem; margin-bottom: 0.6rem; }}
.tt-brand {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.9rem; font-weight: 600;
             letter-spacing: 0.12em; color: var(--text); }}
.tt-brand span {{ color: var(--accent); }}
.tt-pitch {{ color: var(--text-dim); font-size: 0.98rem; margin-top: 0.2rem; max-width: 62rem; }}
.tt-tags {{ margin-top: 0.7rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }}
.tt-tag {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.06em;
           color: var(--text-dim); border: 1px solid var(--panel-border); padding: 0.18rem 0.5rem;
           border-radius: 2px; text-transform: uppercase; }}

/* disclaimer */
.tt-disclaimer {{ border: 1px solid var(--panel-border); border-left: 3px solid var(--warn);
                   background: var(--panel); padding: 0.6rem 0.9rem; border-radius: 2px; margin: 0.8rem 0 1.1rem 0; }}
.tt-disclaimer .tt-label {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; letter-spacing: 0.08em;
                             color: var(--warn); text-transform: uppercase; margin-bottom: 0.25rem; }}
.tt-disclaimer .tt-body {{ color: var(--text); font-size: 0.88rem; line-height: 1.45; }}

/* section labels */
.tt-section-label {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.1em;
                      color: var(--text-dim); text-transform: uppercase; margin: 1.1rem 0 0.4rem 0; }}

/* legend */
.tt-legend {{ display: flex; gap: 1.2rem; flex-wrap: wrap; font-family: 'IBM Plex Mono', monospace;
              font-size: 0.75rem; color: var(--text-dim); margin: 0.3rem 0 0.2rem 0; }}
.tt-swatch {{ display: inline-block; width: 0.7rem; height: 0.7rem; margin-right: 0.4rem;
              vertical-align: middle; border-radius: 1px; }}

/* metric tiles */
.tt-tile {{ border: 1px solid var(--panel-border); border-top: 3px solid var(--accent); background: var(--panel);
            padding: 0.7rem 0.9rem; border-radius: 2px; }}
.tt-tile .tt-tile-label {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.08em;
                            color: var(--text-dim); text-transform: uppercase; }}
.tt-tile .tt-tile-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.65rem; color: var(--text);
                            margin-top: 0.15rem; }}
.tt-tile .tt-tile-unit {{ font-size: 0.95rem; color: var(--text-dim); }}

/* buttons */
.stButton button, .stDownloadButton button {{
    border-radius: 2px !important; border: 1px solid var(--accent) !important;
    background: transparent !important; color: var(--accent) !important;
    font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.04em;
}}
.stButton button:hover, .stDownloadButton button:hover {{
    background: var(--accent) !important; color: var(--bg) !important;
}}

/* radio group -> segmented look */
div[role="radiogroup"] label {{
    font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.03em;
}}

footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached, expensive resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    import torch
    # Deployed app assumes no GPU (Streamlit Community Cloud / HF Spaces free tier).
    return load_model(BEST_CHECKPOINT_PATH, device=torch.device("cpu"))


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
    """Runs once per loaded volume; predicted labels + per-voxel class
    probabilities (needed for the confidence heatmap view) are computed in
    the same forward pass and cached in session_state so neither the plane
    tabs nor the sliders ever re-trigger the model. Uses test-time
    augmentation (horizontal-flip averaging) by default -- roughly doubles
    inference time, which is still well under a second per slice, for
    typically better boundary accuracy."""
    if st.session_state.get("volume_key") != cache_key:
        model, device = get_model()
        with st.spinner("Running segmentation model on all slices (one-time per volume)..."):
            pred_mask, probs = predict_full_volume(image, model, device, return_probs=True, use_tta=True)
        st.session_state["volume_key"] = cache_key
        st.session_state["pred_mask"] = pred_mask
        st.session_state["probs"] = probs
        st.session_state["image"] = image
    return st.session_state["pred_mask"], st.session_state["probs"]


# ---------------------------------------------------------------------------
# Geometry helpers — multi-planar reformatting from the same 3D volume
# ---------------------------------------------------------------------------
PLANES = {"Axial": 2, "Sagittal": 0, "Coronal": 1}


def get_plane_slice(volume_3d, axis, index):
    """volume_3d: (X, Y, Z). Returns a 2D slice oriented with the long (Z)
    axis vertical for sagittal/coronal, matching conventional radiology
    layout; axial is returned as-is."""
    if axis == 2:
        return volume_3d[:, :, index]
    if axis == 0:
        return np.flipud(volume_3d[index, :, :].T)
    return np.flipud(volume_3d[:, index, :].T)


def axis_tumor_profile(pred_mask, axis):
    sum_axes = tuple(a for a in range(3) if a != axis)
    return (pred_mask > 0).sum(axis=sum_axes)


def default_slice_index(pred_mask, axis):
    profile = axis_tumor_profile(pred_mask, axis)
    if profile.max() == 0:
        return pred_mask.shape[axis] // 2
    return int(np.argmax(profile))


PATTERN_NAMES = {1: "dots", 2: "diagonal stripes", 3: "crosshatch"}


def pattern_mask(label, shape):
    """Boolean texture mask distinguishing tumor sub-regions by shape, not
    just hue -- so the overlay stays legible for colorblind users or in
    grayscale print, not only to users who can see red/yellow/blue apart."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    if label == 1:  # necrotic / non-enhancing core -> dots
        period = 8
        cy, cx = yy % period - period // 2, xx % period - period // 2
        return (cy ** 2 + cx ** 2) <= 1.6 ** 2
    if label == 2:  # edema -> diagonal stripes
        period = 8
        return ((xx + yy) % period) < 2
    if label == 3:  # enhancing tumor -> crosshatch
        period = 9
        return ((xx % period) < 2) | ((yy % period) < 2)
    return np.zeros(shape, dtype=bool)


def overlay_rgb(base_gray, label_slice, visible_labels, alpha, use_patterns=False):
    rgb = np.stack([base_gray] * 3, axis=-1)
    for label, color in OVERLAY_COLORS.items():
        if label not in visible_labels:
            continue
        mask = label_slice == label
        for c in range(3):
            rgb[..., c] = np.where(mask, (1 - alpha) * rgb[..., c] + alpha * color[c], rgb[..., c])
        if use_patterns:
            texture = mask & pattern_mask(label, label_slice.shape)
            for c in range(3):
                rgb[..., c] = np.where(texture, rgb[..., c] * 0.35, rgb[..., c])
    return np.clip(rgb, 0, 1)


def normalize_for_display(slice_2d):
    lo, hi = np.percentile(slice_2d, 1), np.percentile(slice_2d, 99)
    if hi <= lo:
        return np.zeros_like(slice_2d)
    return np.clip((slice_2d - lo) / (hi - lo), 0, 1)


def render_header():
    st.markdown(f"""
    <div class="tt-header">
        <div class="tt-brand">TUMOR<span>TRACE</span></div>
        <div class="tt-pitch">TumorTrace draws the exact boundary of a brain tumor on an MRI scan in
        seconds — trained on 369 real glioma patients from the BraTS 2020 challenge.</div>
        <div class="tt-tags">
            <span class="tt-tag">Model · ResNet34-UNet</span>
            <span class="tt-tag">Data · BraTS 2020</span>
            <span class="tt-tag">Inference · CPU</span>
            <span class="tt-tag">Status · Research prototype</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_disclaimer():
    st.markdown(f"""
    <div class="tt-disclaimer">
        <div class="tt-label">Notice — research use only</div>
        <div class="tt-body">{DISCLAIMER}</div>
    </div>
    """, unsafe_allow_html=True)


def render_legend(use_patterns=False):
    label_ids = {"enhancing": 3, "edema": 2, "necrotic": 1}
    swatches = "".join(
        f'<span><span class="tt-swatch" style="background:{REGION_HEX[key]}"></span>{label}'
        f'{f" ({PATTERN_NAMES[label_ids[key]]})" if use_patterns else ""}</span>'
        for key, label in [("enhancing", "Enhancing tumor"), ("edema", "Edema"), ("necrotic", "Necrotic / non-enhancing core")]
    )
    st.markdown(f'<div class="tt-legend">{swatches}</div>', unsafe_allow_html=True)


def render_metric_tile(col, label, value, unit, accent):
    col.markdown(f"""
    <div class="tt-tile" style="border-top-color:{accent}">
        <div class="tt-tile-label">{label}</div>
        <div class="tt-tile-value">{value} <span class="tt-tile-unit">{unit}</span></div>
    </div>
    """, unsafe_allow_html=True)


def build_report_markdown(case_label, volumes, zooms, plane_summary):
    lines = [
        "# TumorTrace segmentation report",
        "",
        DISCLAIMER,
        "",
        f"**Case:** {case_label}",
        f"**Voxel spacing (mm):** {zooms[0]:.2f} x {zooms[1]:.2f} x {zooms[2]:.2f}",
        "",
        "## Tumor sub-region volumes",
        "",
        "| Region | Volume (cm³) |",
        "|---|---|",
        f"| Whole Tumor (WT) | {volumes['WT']:.2f} |",
        f"| Tumor Core (TC) | {volumes['TC']:.2f} |",
        f"| Enhancing Tumor (ET) | {volumes['ET']:.2f} |",
        "",
        "## Largest cross-section per plane",
        "",
        "| Plane | Slice index (largest tumor extent) |",
        "|---|---|",
    ]
    for plane, idx in plane_summary.items():
        lines.append(f"| {plane} | {idx} |")
    lines += [
        "",
        "*Generated by TumorTrace — a research and educational prototype, not a medical device.*",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
render_header()
render_disclaimer()

mode_col, _ = st.columns([2, 3])
with mode_col:
    mode = st.radio("Input", ["Try a sample case", "Upload your own"], horizontal=True, label_visibility="collapsed")

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
        case_label = case_name

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
        case_label = "Uploaded case"
    else:
        st.info("Upload all 4 modalities (T1, T1ce, T2, FLAIR) to run segmentation.")

# ---------------------------------------------------------------------------
# Inference + visualization
# ---------------------------------------------------------------------------
if image is not None:
    pred_mask, probs = run_inference(image, cache_key)
    confidence_map = probs.max(axis=0)  # (H, W, D)

    if st.session_state.get("last_key_for_slider") != cache_key:
        for plane_name, axis in PLANES.items():
            st.session_state[f"slice_{plane_name}"] = default_slice_index(pred_mask, axis)
        st.session_state["last_key_for_slider"] = cache_key

    st.markdown('<div class="tt-section-label">View controls</div>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.4, 1, 1.6])
        with c1:
            base_modality = st.selectbox("Base modality", [m.upper() for m in MODALITIES],
                                          index=MODALITIES.index("flair"))
        with c2:
            view_mode = st.radio("Overlay", ["Segmentation", "Confidence"], horizontal=True)
        with c3:
            alpha = st.slider("Overlay opacity", 0.0, 1.0, OVERLAY_ALPHA, 0.05)

        st.markdown("**Visible sub-regions**")
        r1, r2, r3 = st.columns(3)
        show_necrotic = r1.checkbox("Necrotic / non-enh. core", value=True)
        show_edema = r2.checkbox("Edema", value=True)
        show_enhancing = r3.checkbox("Enhancing tumor", value=True)
        visible_labels = set()
        if show_necrotic:
            visible_labels.add(1)
        if show_edema:
            visible_labels.add(2)
        if show_enhancing:
            visible_labels.add(3)

        use_patterns = st.checkbox(
            "Colorblind-friendly patterns (dots / stripes / crosshatch)", value=False
        )

    tabs = st.tabs(list(PLANES.keys()) + ["3D View"])
    plane_summary = {}
    for tab, (plane_name, axis) in zip(tabs, PLANES.items()):
        with tab:
            num_slices = pred_mask.shape[axis]
            slice_idx = st.slider(f"{plane_name} slice", 0, num_slices - 1, key=f"slice_{plane_name}")
            plane_summary[plane_name] = st.session_state[f"slice_{plane_name}"]

            modality_idx = MODALITIES.index(base_modality.lower())
            raw_slice = normalize_for_display(get_plane_slice(image[modality_idx], axis, slice_idx))
            pred_slice = get_plane_slice(pred_mask, axis, slice_idx)

            col_left, col_right = st.columns(2)
            with col_left:
                with st.container(border=True):
                    st.markdown(f'<div class="tt-section-label">{base_modality} — raw</div>', unsafe_allow_html=True)
                    fig, ax = plt.subplots()
                    fig.patch.set_facecolor("#12181f")
                    ax.imshow(raw_slice, cmap="gray")
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)

            with col_right:
                with st.container(border=True):
                    if view_mode == "Segmentation":
                        st.markdown('<div class="tt-section-label">Predicted tumor overlay</div>', unsafe_allow_html=True)
                        fig, ax = plt.subplots()
                        fig.patch.set_facecolor("#12181f")
                        ax.imshow(overlay_rgb(raw_slice, pred_slice, visible_labels, alpha, use_patterns))
                        ax.axis("off")
                        st.pyplot(fig)
                        plt.close(fig)
                        render_legend(use_patterns)
                    else:
                        st.markdown('<div class="tt-section-label">Model confidence (max softmax prob.)</div>', unsafe_allow_html=True)
                        conf_slice = get_plane_slice(confidence_map, axis, slice_idx)
                        fig, ax = plt.subplots()
                        fig.patch.set_facecolor("#12181f")
                        im = ax.imshow(conf_slice, cmap="inferno", vmin=0, vmax=1)
                        ax.axis("off")
                        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                        cbar.ax.tick_params(colors="#7d8b99", labelsize=7)
                        st.pyplot(fig)
                        plt.close(fig)

            with st.container(border=True):
                st.markdown(f'<div class="tt-section-label">Tumor extent across {plane_name.lower()} slices</div>', unsafe_allow_html=True)
                profile = axis_tumor_profile(pred_mask, axis)
                fig, ax = plt.subplots(figsize=(8, 1.6))
                fig.patch.set_facecolor("#12181f")
                ax.set_facecolor("#12181f")
                ax.fill_between(range(len(profile)), profile, color="#ff5a45", alpha=0.35)
                ax.plot(range(len(profile)), profile, color="#ff5a45", linewidth=1.2)
                ax.axvline(slice_idx, color="#e2a53e", linewidth=1.2)
                ax.set_xlim(0, len(profile) - 1)
                ax.tick_params(colors="#7d8b99", labelsize=7)
                for spine in ax.spines.values():
                    spine.set_color("#263241")
                st.pyplot(fig)
                plt.close(fig)

    with tabs[len(PLANES)]:
        st.markdown('<div class="tt-section-label">Interactive 3D render</div>', unsafe_allow_html=True)
        st.caption(
            "Client-side WebGL (NiiVue) — drag to rotate, scroll to zoom, "
            "use the sliders below the render for cross-sections and tumor opacity."
        )
        volume_cache_key = (cache_key, base_modality)
        if st.session_state.get("viewer3d_key") != volume_cache_key:
            modality_idx = MODALITIES.index(base_modality.lower())
            with st.spinner("Encoding volumes for the 3D viewer..."):
                base_b64, mask_b64 = encode_volumes(image[modality_idx], pred_mask, affine)
            st.session_state["viewer3d_key"] = volume_cache_key
            st.session_state["viewer3d_base_b64"] = base_b64
            st.session_state["viewer3d_mask_b64"] = mask_b64
        html_str = build_viewer_html(
            st.session_state["viewer3d_base_b64"], st.session_state["viewer3d_mask_b64"], height=600
        )
        st.iframe(html_str, height=600)

    st.markdown('<div class="tt-section-label">Tumor volume summary</div>', unsafe_allow_html=True)
    volumes = compute_region_volumes_cm3(pred_mask, zooms)
    t1, t2, t3 = st.columns(3)
    render_metric_tile(t1, "Whole Tumor (WT)", f"{volumes['WT']:.1f}", "cm³", "var(--accent)")
    render_metric_tile(t2, "Tumor Core (TC)", f"{volumes['TC']:.1f}", "cm³", REGION_HEX["necrotic"])
    render_metric_tile(t3, "Enhancing Tumor (ET)", f"{volumes['ET']:.1f}", "cm³", REGION_HEX["enhancing"])

    st.markdown('<div class="tt-section-label">Export</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)

    tmp_path = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    nib.save(nib.Nifti1Image(pred_mask.astype(np.uint8), affine), tmp_path)
    with open(tmp_path, "rb") as f:
        mask_bytes = f.read()
    os.remove(tmp_path)

    with d1:
        st.download_button("Download predicted mask (.nii.gz)", data=mask_bytes,
                            file_name="tumortrace_prediction.nii.gz", mime="application/gzip")
    with d2:
        report_md = build_report_markdown(case_label, volumes, zooms, plane_summary)
        st.download_button("Download report (.md)", data=report_md,
                            file_name="tumortrace_report.md", mime="text/markdown")

st.markdown("---")
st.caption(
    "TumorTrace is a research/educational prototype (see disclaimer above). "
    "Model: ResNet34-encoder U-Net trained on BraTS 2020. Not for clinical use."
)
