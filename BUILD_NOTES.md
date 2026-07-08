# Build Notes

Autonomous build log for TumorTrace. Documents every deviation from the literal
spec and why, per the build prompt's instruction to log edge cases here
instead of pausing to ask.

## Update: retrained on the real BraTS 2020 cohort

Everything below this note originally described a synthetic-data stand-in,
because this build started in a sandbox with no Kaggle credentials and no
GPU. The user later provided Kaggle access, and this machine turned out to
have Apple Silicon (M4 Pro) with an MPS backend that benchmarked ~13x faster
than CPU per training batch — `model.best_available_device()` now picks it
up automatically. With that, the real thing became possible:

- Downloaded the real `awsaf49/brats20-dataset-training-validation` dataset
  (4.2GB) via the Kaggle API.
- `preprocess.py` processed 368 of the 369 real training patients (one,
  `BraTS20_Training_355`, has a known corrupt/misnamed segmentation file —
  see "Bugs found" below — and is skipped automatically rather than aborting
  the whole run), yielding 27,618 slices.
- `train.py` trained for real on MPS: early stopping triggered at **epoch
  18** (not the full 50), taking about 85 minutes wall-clock. Best val WT
  Dice 0.8344.
- `evaluate.py` against the 56 real held-out test patients: **WT Dice
  0.904, TC 0.821, ET 0.768** — all of which land at or above the honest
  target ranges stated in the README (0.80–0.88 / 0.70–0.80 / 0.65–0.75),
  not just within them. `results/metrics_table.md`,
  `results/qualitative_examples.png`, and `samples/*.npz` were all
  regenerated from this real checkpoint and now show real patient data
  (BraTS is released for research use, so bundling de-identified sample
  volumes from it is within the dataset's terms).

The synthetic-data path documented below is kept as-is: it's exactly what
happened first, it's still how `dev_tools/make_synthetic_brats.py` and the
rest of this repo's code paths were originally exercised end-to-end without
real data access, and it remains a legitimate fallback for anyone re-running
this build in a sandbox without Kaggle/GPU access.

## The one big deviation: synthetic data instead of real BraTS 2020 (historical — see update above)

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
  handful of epochs (15, not 50) on 24 synthetic patients (not 369 real
  ones). It demonstrates the full training/checkpointing/early-stopping
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

## Bugs found and fixed while wiring up the real training run

Running the full pipeline end-to-end on synthetic data (see above) surfaced
three real bugs that a read-through would not have caught:

1. **`monai.metrics.compute_hausdorff_distance` rejects `numpy.float32`
   spacing values** (`evaluate.py`) — `nibabel`'s `header.get_zooms()`
   returns a tuple of `numpy.float32`, and MONAI's `prepare_spacing` only
   accepts plain Python numeric types. Fixed by casting each element to
   `float()` before passing `spacing=`.
2. **`inference.load_model()` built the model with
   `encoder_weights="imagenet"`**, which triggers a Hugging Face Hub
   download/lookup for pretrained ResNet34 weights — completely wasted
   work, since `load_state_dict()` immediately overwrites every weight with
   the trained checkpoint two lines later. Worse, if that network call
   stalls (as it did once here), `load_model()` hangs with no error, which
   silently hangs the whole app on first load. Fixed by passing
   `encoder_weights=None` in `load_model()` specifically (training's
   `build_model()` call in `train.py` is untouched and still uses
   `"imagenet"`, which is correct there).
3. **matplotlib defaults to the `macosx` GUI backend on this machine**,
   which requires the main thread. Streamlit runs the user script on a
   worker thread, so the first `plt.subplots()` call hung silently (near-0%
   CPU, page stuck on Streamlit's initial loading skeleton forever, no
   exception surfaced anywhere). Fixed by calling `matplotlib.use("Agg")`
   before `import matplotlib.pyplot` in both `app.py` and `evaluate.py` —
   the standard fix for matplotlib-in-a-web-server. This one cost the most
   debugging time because it looks identical to a networking/proxy issue
   from the outside (compare with the next section).

## Bugs found running against the real BraTS 2020 data specifically

Real data surfaced problems synthetic data structurally couldn't, since the
synthetic generator never produced anything nested, malformed, or large
enough to expose them:

4. **`discover_patients()` only checked immediate subdirectories.** The real
   Kaggle zip extracts as `BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/
   BraTS20_Training_XXX/`, two levels deeper than the flat layout the
   synthetic generator produces. Fixed by making `discover_patients()`
   recurse up to 3 levels down when no patient folders are found directly,
   so `--raw_dir data/raw` works regardless of how the archive nests things
   — this is a real, permanent improvement, not a one-off workaround (see
   the recursion depth guard and `_is_patient_dir` split I added).
5. **One real patient aborted the entire preprocessing run.**
   `BraTS20_Training_355`'s segmentation file ships as
   `W39_1998.09.19_Segm.nii` instead of the standard `*_seg.nii` naming —
   a documented anomaly in the public BraTS20 dump, not something I
   introduced. `process_patient()` correctly raised on it, but `main()`'s
   loop had no error handling, so one bad patient out of 369 killed
   progress on all the others. Fixed by catching and logging per-patient
   failures and continuing; the run summary now reports how many patients
   succeeded vs. were skipped and why. Final count: 368/369 processed.
6. **`train.py`/`inference.py` never selected MPS.** Both hardcoded `"cuda"
   if available else "cpu"`, silently ignoring Apple Silicon's GPU backend.
   On synthetic data (small, ran once) this didn't matter enough to notice.
   On the real 368-patient dataset it mattered a lot: benchmarked MPS at
   ~13x faster than CPU per batch (162ms vs 2089ms), which is the
   difference between the real 50-epoch run finishing in about an hour and
   it taking most of a day. Added `model.best_available_device()`
   (cuda > mps > cpu) and used it in both places.
7. **`DataLoader(num_workers=4)` silently stalled** on the real dataset in
   this environment — the process sat at near-0% CPU for 10+ minutes with
   no error, no crash, just nothing happening (multiprocessing worker
   processes appear to not start correctly in this sandbox). This is
   different from bug #6: it's not a matter of speed, it's a hang. Confirmed
   by isolating `build_datasets()` in the same environment (a clean 38.7s,
   not stuck) and by testing `num_workers=0` (worked correctly, actively
   consuming CPU throughout). The real 50-epoch run used `num_workers=0`.
   If you're running this somewhere without that restriction, `num_workers`
   in the 2-4 range should load data faster during training.

## A note on how app.py was actually verified

The browser-automation tool available in this environment could not
render the Streamlit app: even a trivial one-line `st.title("hello")`
smoke-test app hung on Streamlit's loading skeleton indefinitely with the
same near-0%-CPU signature as bug #3 above, before that bug was found —
which is what made bug #3 hard to isolate (identical symptom, two
different causes: one was my bug, one is an environment limitation).

I confirmed the root cause precisely rather than assuming it: from inside
the browser tab, a raw `new WebSocket("ws://localhost:8600/_stcore/stream")`
never leaves `readyState 0` (CONNECTING) — no open, error, or close event,
ever — while `new WebSocket("wss://echo.websocket.org/")` from the same tab
opens and echoes a message in under a second. So this sandbox's browser
tool can make plain HTTP requests to localhost (the static `site/index.html`
and Streamlit's own JS/CSS assets load fine) but cannot complete a
WebSocket handshake to localhost specifically. That's a hard restriction
with no code-side fix — Streamlit's entire rendering model depends on that
connection, so no app.py change could route around it.

Given that, verification proceeded in two complementary ways, both with
zero browser/WebSocket dependency:

1. **`streamlit.testing.v1.AppTest`** (runs the script directly through
   Python, no browser involved) — drove initial load with the default
   sample case, all three plane tabs, the Segmentation→Confidence overlay
   toggle, dragging the axial slider, unchecking a sub-region visibility
   box, and confirmed both download buttons render. Zero exceptions.
2. **Direct visual rendering** — called `app.py`'s own functions
   (`get_plane_slice`, `overlay_rgb`, `normalize_for_display`,
   `axis_tumor_profile`, `default_slice_index`) and `inference.py`'s
   `predict_full_volume` directly against the bundled samples, saved the
   exact same matplotlib figures the app would show, and inspected them as
   images. This confirmed, by actually looking: all three plane
   reformats are anatomically coherent (not garbled/mirrored by the
   transpose/flip logic); the segmentation overlay colors are correct;
   the confidence heatmap is high everywhere except right at class
   boundaries (the expected pattern for a calibrated model, not a bug);
   the region-visibility toggle actually removes the right color when a
   checkbox is unchecked; the tumor-extent profile's amber marker lines
   up with the peak of the curve; all four modality selections render
   with visibly distinct contrast (matching each modality's synthetic
   intensity profile); and the opacity slider scales the overlay smoothly
   from 0.0 to 0.8. Also verified: the NIfTI mask download round-trips
   through `nib.load` with the original 240×240×155 geometry and label
   set intact, the markdown report's content matches the true computed
   volumes/slice indices, and the "Upload your own" code path (temp-file
   writing + `preprocess.load_patient_volumes`) loads a real synthetic
   patient directory correctly.

That combination — zero exceptions under every interaction AppTest can
drive, plus actually looking at the pixels those interactions produce —
is the basis for calling the app verified end-to-end despite never seeing
it live in a browser. If you have a normal (non-sandboxed) browser,
`streamlit run app.py` and click through it yourself too.

## train.ipynb: made it actually portable, then actually ran it

`train.ipynb` was originally written by hand to mirror `train.py`, but
never executed -- it had two Colab-only cells (`from google.colab import
files`, `!git clone ... && %cd tumortrace`) that would fail outright on a
local Jupyter kernel. Added an `IN_COLAB` detection cell at the top and
guarded both, and switched the Kaggle-auth cell to check for
`~/.kaggle/access_token` first (works in both environments) before falling
back to the Colab file-upload widget. Also fixed the same two things real
data had already taught me for the plain scripts: `raw_dir` now points at
the `BraTS2020_TrainingData` branch specifically instead of the flat zip
root (avoiding the unlabeled validation set), and the preprocessing cell
catches and logs per-patient failures instead of letting one bad patient
(`BraTS20_Training_355`) abort the whole cell.

Then I actually ran it, rather than trusting that it would work because the
underlying functions were already proven via `train.py`. Executed via
`jupyter nbconvert --execute` from a temporary copy placed in the repo root
(nbconvert's kernel defaults to the *notebook's own* directory as its
working directory, not the caller's -- the first attempt failed with
`ModuleNotFoundError: No module named 'preprocess'` until I figured that
out). The copy skipped the redundant re-download (data was already on disk
from the real run) and redirected the train/evaluate cells to scratch
paths (`/tmp/tumortrace_verify_checkpoint.pt`, a throwaway 1-epoch run) so
it couldn't clobber the real checkpoint or results. Every cell ran
correctly end to end: Colab detection, the credential check, preprocessing
(368/369 patients, 27,618 slices -- an exact match with the real script-driven
run), one real training epoch (val WT Dice 0.799, consistent with epoch 1
of the real run), and a real `evaluate.py` invocation producing real
Dice/HD95/sensitivity/specificity numbers. Scratch artifacts and the
temporary notebook copy were deleted afterward; the committed `train.ipynb`
itself was never touched by the verification run.

## Second elevation pass: real data, 3D viewer, TTA, accessibility

After the first review pass, the user asked to (1) get real Kaggle credentials
and actually train on real BraTS 2020 data, and (2) push further on "maximize
perceived complexity and functionality" plus a real 3D brain visualization
with region highlighting and cross-sections, "research looking style."

- **3D viewer (`viewer3d.py`)**: evaluated PyVista/stpyvista, Plotly
  `go.Volume`/`go.Isosurface`, and NiiVue. Picked **NiiVue**
  (niivue.github.io) specifically because it's a real tool used in published
  neuroimaging research (not a generic 3D library repurposed for this), and
  because it renders 100% client-side via WebGL — no server-side VTK/OSMesa/
  xvfb dependency, which is what would make PyVista risky to deploy on
  Streamlit Community Cloud's free containers. Embedded via `st.iframe`
  (not `st.components.v1.html`, which `AppTest` flagged as past its
  announced removal window) as a 4th "3D View" tab alongside the existing
  plane tabs, with the cross-section and opacity sliders built into the
  HTML component itself — a component iframe can't be wired to external
  Streamlit widgets without a custom bidirectional protocol, so those
  controls live inside the generated HTML, not as separate `st.slider` calls.
  Getting here took real debugging, not just picking a library and moving
  on: the CDN URL/version I first guessed (`niivue@0.44.0/dist/niivue.esm.min.js`)
  doesn't exist; the real ESM entry point has bare-specifier imports
  (`gl-matrix`, etc.) that need a full import-map to resolve; the working
  path is the UMD build (`niivue.umd.js`) via a plain `<script>` tag, which
  self-contains all dependencies. Loading 3 separately-colored region
  volumes (necrotic/edema/enhancing as distinct hues) in 3D render mode also
  didn't composite correctly (colors silently disappeared) even though the
  volumes loaded with correct data — rather than debug niivue's multi-overlay
  3D blending further, I fell back to one combined tumor-region volume with
  a single highlight color in the 3D view specifically, since the 2D plane
  tabs already give the full 3-color WT/TC/ET breakdown in detail. Verified
  the whole thing by generating the exact production HTML output and
  loading it in a real browser before wiring it into `app.py` at all.
- **Test-time augmentation** (`inference.py`): averages softmax
  probabilities from each slice and its horizontal flip before argmax.
  Deliberately *not* a 2.5D (multi-slice-context) upgrade, which the spec's
  exact `in_channels=4` architecture would have to change to accommodate —
  TTA is a real accuracy improvement that doesn't touch the pinned
  architecture. Verified: TTA and non-TTA predictions agree 99.99% of the
  time, differing only at ambiguous class-boundary voxels, which is exactly
  what TTA is supposed to affect.
- **Colorblind-accessible overlay patterns** (`app.py`): optional dot/
  diagonal-stripe/crosshatch texture per tumor sub-region, toggleable,
  so the overlay stays legible without relying on hue alone.
- **`requirements.txt` pinned** to exact versions verified working in this
  venv (was `>=` ranges) — `albumentations` in particular has had breaking
  API changes across major versions, so an unpinned install could silently
  build something different from what was tested here.

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
