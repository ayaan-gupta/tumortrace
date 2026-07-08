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
