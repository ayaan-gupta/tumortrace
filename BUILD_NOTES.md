# Build Notes

Autonomous build log for TumorTrace. Documents every deviation from the literal
spec and why, per the build prompt's instruction to log edge cases here
instead of pausing to ask.

## The one big deviation: synthetic data instead of real BraTS 2020

This build ran in a sandboxed environment with **no Kaggle credentials and no
GPU**. Downloading the real ~7GB BraTS20 Kaggle dataset and running 50 epochs
of GPU training were both infeasible here. Rather than ship a repo with dead
code nobody had ever run, I built a small script,
[`dev_tools/make_synthetic_brats.py`](dev_tools/make_synthetic_brats.py),
that generates synthetic patient volumes in **exactly** the real BraTS
directory/filename/label/geometry convention (240×240×155, raw labels
`{0,1,2,4}`, skull-stripped-style zero background, nested ellipsoid
"tumors" with modality-specific intensity shifts so there's real learnable
multi-modal signal). I ran the full pipeline — `preprocess.py` → `train.py`
→ `evaluate.py` → `make_samples.py` → `app.py` — against this synthetic data
end-to-end, so every module in this repo is exercised by a real run, not just
reviewed by eye.

**What this means concretely:**
- `checkpoints/best_model.pt` is trained on synthetic data only, for a
  handful of epochs (not 50) on a couple dozen synthetic patients (not 369
  real ones). It demonstrates the full training/checkpointing/early-stopping
  machinery works, but it has **not** learned real glioma tissue patterns.
- `results/metrics_table.md` and `results/qualitative_examples.png` reflect
  performance on synthetic data. The Dice scores you'll see there are
  higher than the realistic BraTS targets stated in the README, because
  the synthetic "tumors" are geometrically clean ellipsoids — an easier
  problem than real, irregular glioma tissue. **Do not read them as a claim
  about real-world performance;** the README's target ranges (WT 0.80–0.88,
  TC 0.70–0.80, ET 0.65–0.75) are the honest expectation for a real training
  run, sourced from the published literature on this class of approach.
- `samples/*.npz` are synthetic demo cases, not real patient scans (no real
  patient data is bundled in this repo).
- To get a real, clinically-relevant model, follow the README's "Run it
  yourself" section instead: it downloads the real Kaggle dataset and runs
  `train.py` for the full 50 epochs (ideally on a Colab T4 via
  `train.ipynb`). Swap in the resulting `checkpoints/best_model.pt` and
  re-run `evaluate.py` and `make_samples.py` before deploying for real.

`dev_tools/` is not part of the spec's required file list (§10) — it's a
sandbox-only aid so this repo's code paths are demonstrably correct, kept
separate so it's obviously not part of the production interface.

## Product elevations beyond the base spec

The user asked, mid-build, not to leave out anything that could elevate the
product, and for a visual identity that doesn't read as generic AI-generated
UI (no gradient hero banners, no glassmorphic rounded cards). `app.py` ended
up meaningfully beyond the §9 spec as a result:

- **Custom "radiology workstation" visual identity**: flat bordered panels
  (`st.container(border=True)`, re-skinned via CSS to sharp corners + a
  charcoal/coral palette), IBM Plex Mono for technical labels, rectangular
  outlined tags instead of rounded pills, and a `.streamlit/config.toml`
  dark theme so native widgets (sliders, buttons, checkboxes) match without
  fighting Streamlit's internal DOM structure.
- **Multi-planar reformatting**: Axial / Sagittal / Coronal tabs, each with
  its own independent slice slider and its own "largest tumor cross-section"
  default, all reslicing the *same* cached 3D prediction — no extra
  inference cost.
- **Model-confidence heatmap mode**: a toggle next to the segmentation
  overlay that renders per-voxel max-softmax-probability instead of the
  label overlay, so a user can see where the model is uncertain, not just
  what it predicted. `inference.predict_full_volume(..., return_probs=True)`
  computes this in the *same* forward pass as the segmentation labels (one
  inference run per volume, per the spec's performance requirement — the
  confidence feature does not add a second model pass).
- **Overlay controls**: opacity slider and per-sub-region visibility
  checkboxes (hide edema to see the core more clearly, etc.), not present in
  the base spec.
- **Tumor-extent profile**: a small area chart per plane showing tumor voxel
  count across every slice index, with the current slice marked — lets a
  user jump straight to where the tumor is largest instead of scrubbing
  blindly.
- **Downloadable markdown report** alongside the NIfTI mask download:
  per-region volumes, voxel spacing, and the largest-cross-section slice
  index per plane, with the disclaimer restated in the file itself so it
  travels with the artifact if shared.

## Standalone product site (site/index.html)

Midway through the build the user asked for a genuinely designed marketing
front-door for the tool — three.js, animation, editorial typography, dark
theme — in the register of premium tech-startup sites (they attached
reference screenshots), explicitly *not* the Streamlit app's default look.
`site/index.html` is a self-contained static page (three.js loaded via
import-map CDN, Fraunces/Inter/IBM Plex Mono via Google Fonts, no build
step) with a hero built around the product's actual geometry, not generic
decoration: a wireframe icosahedron ("head") with three orbiting rings
representing the axial/sagittal/coronal imaging planes, and a highlighted
"tumor" node with accent-colored connecting lines. It links out to the
Streamlit app (`appUrl` in the config block at the bottom of the file —
update this after deploying) and to the GitHub repo (`githubUrl`, same
block). The brand accent (coral-red, `#ff5a45`) was then carried back into
`app.py`'s "workstation" theme so the two surfaces read as one product
instead of two different palettes.

I browser-tested this myself end-to-end (desktop viewport, scroll-reveal
animations, three.js render, all section content, console errors) rather
than asking the user to — found and fixed one real bug this way: a stale
closure in the animated console-log loop (`step` had already been
incremented by the time a `setTimeout` callback referenced `lineEls[step]`,
which threw once the loop reached the last line). Mobile-viewport testing
via the browser tool's window-resize was unreliable in this sandbox (the
resize call reported success but `window.innerWidth` never changed), so the
`@media` breakpoints are verified by source inspection (standard, simple
rules) rather than a live narrow-viewport screenshot — worth a manual check
on a real phone before shipping.

## Other decisions / edge cases

- **Python 3.11 used instead of 3.10.** Spec says "3.10+"; 3.11 was the
  newest available interpreter in this environment and is within spec.
- **`torch.autocast`/`GradScaler` guarded by `device.type == "cuda"`.** AMP
  is a no-op on CPU (no half-precision benefit and some ops are unsupported),
  so `train.py` only enables it when a CUDA device is present — exactly the
  Colab T4 scenario the spec calls out, while still working correctly (just
  without AMP) if someone runs `train.py` locally on CPU.
- **Patient-level split derived once and cached to `data/processed/split_patients.json`.**
  `evaluate.py` and `make_samples.py` both read this file rather than
  re-deriving the split, guaranteeing the "held-out" test set used for
  reporting is identical to the one training never saw.
- **Evaluation metrics computed patient-level in 3D**, not slice-level in 2D.
  The spec's Dice/HD95/sensitivity/specificity targets are standard BraTS
  metrics, which are defined over whole 3D volumes per patient; `evaluate.py`
  reconstructs each test patient's full volume from per-slice predictions
  (mirroring `inference.py`) before scoring, then averages across patients.
- **HD95 excludes patient/region pairs where only one of {prediction, ground
  truth} is empty** (Hausdorff distance is undefined between an empty set and
  a non-empty one). Both-empty pairs score 0 (perfect agreement); the
  ill-defined case is dropped from the average rather than silently
  poisoning it with an arbitrary sentinel value.
- **Modality filename matching is suffix-based** (`*_t1.nii.gz`,
  `*_t1ce.nii.gz`, etc., matched with anchored regexes so `t1` doesn't
  accidentally match `t1ce`), not tied to the exact BraTS naming scheme —
  this is what makes `preprocess.py` source-agnostic per the spec's
  requirement to support both the Kaggle BraTS20 dump and a
  Medical-Segmentation-Decathlon-derived export.
- **Uploaded volumes in `app.py`** are saved to a temp directory using fixed
  filenames (`upload_t1.nii.gz`, etc.) rather than the user's original
  filenames, so they always match the suffix convention regardless of what
  the user named their files.
- **`checkpoints/best_model.pt` is committed as a plain file, not via Git
  LFS.** The spec's 200MB ceiling assumes a real ResNet34 encoder checkpoint,
  which comfortably fits GitHub's raw file-size limits without LFS; LFS
  should only be introduced if a future checkpoint variant grows past that.
