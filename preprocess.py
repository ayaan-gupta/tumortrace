"""Source-agnostic BraTS preprocessing.

Expects `data_raw_dir` to contain one sub-directory per patient, each holding
five NIfTI volumes whose filenames end in `_t1`, `_t1ce`, `_t2`, `_flair`,
and `_seg` (any prefix is accepted, so this works whether the directory was
populated by the Kaggle BraTS20 dump or a renamed Medical Segmentation
Decathlon export). Produces 2D axial slice pairs as .npy files for fast
per-epoch loading during training, plus a patient-level train/val/test split.
"""
import argparse
import json
import os
import re
from glob import glob

import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split

from constants import (
    CROP_SIZE,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    HARD_NEGATIVE_FRACTION,
    LABEL_MAP,
    MODALITIES,
    RANDOM_STATE,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
)

_MODALITY_PATTERNS = {
    mod: re.compile(rf".*_{mod}\.nii(\.gz)?$", re.IGNORECASE) for mod in MODALITIES
}
_SEG_PATTERN = re.compile(r".*_seg\.nii(\.gz)?$", re.IGNORECASE)


def _find_file(patient_dir, pattern):
    for fname in os.listdir(patient_dir):
        if pattern.match(fname):
            return os.path.join(patient_dir, fname)
    return None


def discover_patients(data_raw_dir):
    """Return {patient_id: patient_dir} for every sub-directory that has all
    4 modality volumes (segmentation mask is optional, since inference-time
    patient directories won't have one)."""
    patients = {}
    for entry in sorted(os.listdir(data_raw_dir)):
        patient_dir = os.path.join(data_raw_dir, entry)
        if not os.path.isdir(patient_dir):
            continue
        modality_paths = {
            mod: _find_file(patient_dir, pat) for mod, pat in _MODALITY_PATTERNS.items()
        }
        if all(modality_paths.values()):
            patients[entry] = patient_dir
    return patients


def zscore_nonzero(volume):
    """Per-volume z-score normalization using only nonzero (brain) voxels."""
    nonzero = volume[volume != 0]
    mean = nonzero.mean() if nonzero.size else 0.0
    std = nonzero.std() if nonzero.size else 1.0
    return (volume - mean) / (std + 1e-8)


def remap_labels(seg):
    remapped = np.zeros_like(seg, dtype=np.uint8)
    for raw_label, new_label in LABEL_MAP.items():
        remapped[seg == raw_label] = new_label
    return remapped


def center_crop_or_pad_2d(img, size=CROP_SIZE):
    """Center-crop (if larger) or zero-pad (if smaller) a 2D array to size x size."""
    h, w = img.shape
    out = np.zeros((size, size), dtype=img.dtype)

    # Source crop window
    src_y0 = max((h - size) // 2, 0)
    src_x0 = max((w - size) // 2, 0)
    src_h = min(h, size)
    src_w = min(w, size)

    # Destination placement window (centers a smaller source in a larger canvas)
    dst_y0 = max((size - h) // 2, 0)
    dst_x0 = max((size - w) // 2, 0)

    out[dst_y0:dst_y0 + src_h, dst_x0:dst_x0 + src_w] = (
        img[src_y0:src_y0 + src_h, src_x0:src_x0 + src_w]
    )
    return out


def uncrop_or_pad_2d(cropped, orig_h, orig_w, size=CROP_SIZE, fill_value=0):
    """Exact inverse of center_crop_or_pad_2d: places a size x size array back
    into an (orig_h, orig_w) canvas (filled with `fill_value` outside the
    original crop window). Used at inference time to map predictions back to
    native BraTS geometry (240x240). `fill_value` defaults to 0 (background
    label / zero probability); pass 1.0 when uncropping the background-class
    probability channel so voxels outside the model's field of view read as
    "certainly background" rather than "certainly nothing"."""
    out = np.full((orig_h, orig_w), fill_value, dtype=cropped.dtype)

    src_y0 = max((orig_h - size) // 2, 0)
    src_x0 = max((orig_w - size) // 2, 0)
    src_h = min(orig_h, size)
    src_w = min(orig_w, size)

    dst_y0 = max((size - orig_h) // 2, 0)
    dst_x0 = max((size - orig_w) // 2, 0)

    out[src_y0:src_y0 + src_h, src_x0:src_x0 + src_w] = (
        cropped[dst_y0:dst_y0 + src_h, dst_x0:dst_x0 + src_w]
    )
    return out


def load_patient_volumes(patient_dir, load_seg=True):
    """Load, normalize, and stack the 4 modalities for one patient.

    Returns:
        image: float32 array (4, H, W, D), z-scored per modality on nonzero voxels
        seg: uint8 array (H, W, D) remapped to {0,1,2,3}, or None if load_seg=False
             or no segmentation file is present (e.g. inference on new uploads).
        zooms: voxel spacing tuple from the NIfTI header (for volume calculations)
        affine: the reference NIfTI affine (for re-saving predictions)
    """
    modality_paths = {
        mod: _find_file(patient_dir, pat) for mod, pat in _MODALITY_PATTERNS.items()
    }
    missing = [m for m, p in modality_paths.items() if p is None]
    if missing:
        raise FileNotFoundError(f"Missing modalities {missing} in {patient_dir}")

    channels = []
    ref_img = None
    for mod in MODALITIES:
        nii = nib.load(modality_paths[mod])
        if ref_img is None:
            ref_img = nii
        data = nii.get_fdata(dtype=np.float32)
        channels.append(zscore_nonzero(data))
    image = np.stack(channels, axis=0)  # (4, H, W, D)

    seg = None
    if load_seg:
        seg_path = _find_file(patient_dir, _SEG_PATTERN)
        if seg_path is not None:
            seg_data = nib.load(seg_path).get_fdata(dtype=np.float32).astype(np.int16)
            seg = remap_labels(seg_data)

    return image, seg, ref_img.header.get_zooms()[:3], ref_img.affine


def extract_slices(image, seg, rng, hard_negative_fraction=HARD_NEGATIVE_FRACTION):
    """Given a full (4,H,W,D) volume and (H,W,D) mask, return the list of axial
    slice indices to keep: every tumor-positive slice, plus a random sample of
    `hard_negative_fraction` of the tumor-free slices."""
    num_slices = image.shape[-1]
    tumor_slices, empty_slices = [], []
    for idx in range(num_slices):
        if seg is not None and np.any(seg[:, :, idx] > 0):
            tumor_slices.append(idx)
        else:
            empty_slices.append(idx)

    n_hard_neg = int(round(len(empty_slices) * hard_negative_fraction))
    hard_negatives = list(rng.choice(empty_slices, size=n_hard_neg, replace=False)) if n_hard_neg else []
    return sorted(tumor_slices + hard_negatives)


def process_patient(patient_id, patient_dir, out_dir, rng):
    image, seg, _, _ = load_patient_volumes(patient_dir, load_seg=True)
    if seg is None:
        raise ValueError(f"Patient {patient_id} has no segmentation mask; skipping.")

    slice_indices = extract_slices(image, seg, rng)
    for idx in slice_indices:
        img_slice = np.stack(
            [center_crop_or_pad_2d(image[c, :, :, idx]) for c in range(image.shape[0])],
            axis=0,
        ).astype(np.float32)
        mask_slice = center_crop_or_pad_2d(seg[:, :, idx]).astype(np.uint8)

        np.save(os.path.join(out_dir, f"image_{patient_id}_{idx}.npy"), img_slice)
        np.save(os.path.join(out_dir, f"mask_{patient_id}_{idx}.npy"), mask_slice)
    return len(slice_indices)


def split_patients(patient_ids):
    train_ids, rest_ids = train_test_split(
        sorted(patient_ids), train_size=TRAIN_FRAC, random_state=RANDOM_STATE
    )
    val_size = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    val_ids, test_ids = train_test_split(
        rest_ids, train_size=val_size, random_state=RANDOM_STATE
    )
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", default=DATA_RAW_DIR)
    parser.add_argument("--out_dir", default=DATA_PROCESSED_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    patients = discover_patients(args.raw_dir)
    if not patients:
        raise SystemExit(
            f"No patient folders with all 4 modalities + seg found under {args.raw_dir}. "
            "See README 'Run it yourself' for dataset download instructions."
        )

    splits = split_patients(list(patients.keys()))
    with open(os.path.join(args.out_dir, "split_patients.json"), "w") as f:
        json.dump(splits, f, indent=2)

    rng = np.random.RandomState(RANDOM_STATE)
    total_slices = 0
    for patient_id, patient_dir in patients.items():
        n = process_patient(patient_id, patient_dir, args.out_dir, rng)
        total_slices += n
        print(f"[preprocess] {patient_id}: {n} slices")

    print(f"[preprocess] Done. {len(patients)} patients, {total_slices} slices saved to {args.out_dir}")
    print(f"[preprocess] Split sizes: "
          f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")


if __name__ == "__main__":
    main()
