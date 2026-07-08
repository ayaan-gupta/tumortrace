"""SANDBOX-ONLY helper — NOT part of the TumorTrace spec's deliverable file list.

Generates a small synthetic dataset shaped exactly like BraTS 2020 (same
directory layout, same filename suffixes, same 240x240x155 volume geometry,
same {0,1,2,4} raw label convention) so the *real* preprocess.py / train.py /
evaluate.py / inference.py / app.py code paths can be exercised end-to-end
without a Kaggle account or GPU. See BUILD_NOTES.md for why this exists.

The synthetic "brain" is a filled ellipsoid of smooth correlated noise; the
synthetic "tumor" is a nested set of ellipsoids (edema shell / enhancing rim /
necrotic core) with modality-specific intensity shifts so there is genuine,
learnable multi-modal signal (not just noise) for the model to pick up on.
This is a stand-in for real glioma data ONLY — do not use any artifact
downstream of this script to make claims about real-world model performance.
"""
import argparse
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

SHAPE = (240, 240, 155)
AFFINE = np.diag([1.0, 1.0, 1.0, 1.0])  # 1mm isotropic, matching real BraTS


def _ellipsoid_mask(shape, center, radii):
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    val = (
        ((zz - center[0]) / radii[0]) ** 2
        + ((yy - center[1]) / radii[1]) ** 2
        + ((xx - center[2]) / radii[2]) ** 2
    )
    return val <= 1.0


def _smooth_noise(shape, rng, sigma=6.0):
    noise = rng.standard_normal(shape).astype(np.float32)
    return gaussian_filter(noise, sigma=sigma)


def make_patient(patient_id, out_dir, rng):
    patient_dir = os.path.join(out_dir, patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    brain_center = (SHAPE[0] // 2, SHAPE[1] // 2, SHAPE[2] // 2)
    brain_radii = (85, 95, 60)
    brain_mask = _ellipsoid_mask(SHAPE, brain_center, brain_radii)

    tumor_center = tuple(
        brain_center[i] + rng.integers(-35, 35) for i in range(3)
    )
    base_r = rng.uniform(12, 22)
    necrotic_radii = (base_r * 0.4, base_r * 0.4, base_r * 0.3)
    enhancing_radii = (base_r * 0.7, base_r * 0.7, base_r * 0.55)
    edema_radii = (base_r * 1.8, base_r * 1.8, base_r * 1.4)

    necrotic_mask = _ellipsoid_mask(SHAPE, tumor_center, necrotic_radii)
    enhancing_full = _ellipsoid_mask(SHAPE, tumor_center, enhancing_radii)
    edema_full = _ellipsoid_mask(SHAPE, tumor_center, edema_radii) & brain_mask

    enhancing_mask = enhancing_full & ~necrotic_mask
    edema_mask = edema_full & ~enhancing_full

    seg = np.zeros(SHAPE, dtype=np.uint8)
    seg[edema_mask] = 2
    seg[enhancing_mask] = 4  # raw BraTS label for enhancing tumor
    seg[necrotic_mask] = 1

    # Per-modality base tissue contrast + tumor-region intensity shifts,
    # loosely mimicking real modality behavior (T1ce enhances the rim,
    # FLAIR/T2 light up edema, T1 is relatively hypointense in the necrotic core).
    modality_params = {
        "t1": dict(base=1.0, necrotic=-0.6, enhancing=0.1, edema=0.05),
        "t1ce": dict(base=1.0, necrotic=-0.3, enhancing=1.2, edema=0.1),
        "t2": dict(base=1.0, necrotic=0.3, enhancing=0.2, edema=0.9),
        "flair": dict(base=1.0, necrotic=0.2, enhancing=0.3, edema=1.1),
    }

    for modality, params in modality_params.items():
        tissue = 100 * params["base"] + 15 * _smooth_noise(SHAPE, rng, sigma=8)
        tissue += 6 * rng.standard_normal(SHAPE).astype(np.float32)  # voxel-level texture
        tissue[necrotic_mask] += 100 * params["necrotic"]
        tissue[enhancing_mask] += 100 * params["enhancing"]
        tissue[edema_mask] += 100 * params["edema"]
        tissue = tissue * brain_mask  # skull-strip: exactly zero outside the brain
        tissue = np.clip(tissue, 0, None).astype(np.float32)

        nib.save(nib.Nifti1Image(tissue, AFFINE),
                  os.path.join(patient_dir, f"{patient_id}_{modality}.nii.gz"))

    nib.save(nib.Nifti1Image(seg, AFFINE), os.path.join(patient_dir, f"{patient_id}_seg.nii.gz"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default="data/raw")
    parser.add_argument("--n_patients", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(1, args.n_patients + 1):
        patient_id = f"SynthBraTS_{i:03d}"
        make_patient(patient_id, args.out_dir, rng)
        print(f"[synthetic] generated {patient_id}")


if __name__ == "__main__":
    main()
