"""PyTorch Dataset over pre-extracted 2D BraTS slice pairs (see preprocess.py)."""
import json
import os
from glob import glob

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset

from constants import DATA_PROCESSED_DIR


def build_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ElasticTransform(p=0.2),
        A.RandomBrightnessContrast(p=0.3),
    ])


def load_split_patients(processed_dir=DATA_PROCESSED_DIR):
    with open(os.path.join(processed_dir, "split_patients.json")) as f:
        return json.load(f)


class BraTSSliceDataset(Dataset):
    """Yields (image, mask) pairs: image is (4, H, W) float32, mask is (H, W) int64."""

    def __init__(self, processed_dir, patient_ids, augment=False):
        self.processed_dir = processed_dir
        self.augment = augment
        self.transform = build_train_transform() if augment else None

        self.image_paths = []
        for patient_id in patient_ids:
            self.image_paths.extend(
                sorted(glob(os.path.join(processed_dir, f"image_{patient_id}_*.npy")))
            )
        if not self.image_paths:
            raise RuntimeError(
                f"No slices found for the given patient_ids in {processed_dir}. "
                "Did you run preprocess.py first?"
            )

    def __len__(self):
        return len(self.image_paths)

    def _mask_path(self, image_path):
        fname = os.path.basename(image_path)
        return os.path.join(self.processed_dir, fname.replace("image_", "mask_", 1))

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.load(image_path)               # (4, H, W)
        mask = np.load(self._mask_path(image_path))  # (H, W)

        if self.transform is not None:
            image_hwc = np.transpose(image, (1, 2, 0))
            augmented = self.transform(image=image_hwc, mask=mask)
            image = np.transpose(augmented["image"], (2, 0, 1))
            mask = augmented["mask"]

        image = np.ascontiguousarray(image, dtype=np.float32)
        mask = np.ascontiguousarray(mask, dtype=np.int64)
        return torch.from_numpy(image), torch.from_numpy(mask)


def build_datasets(processed_dir=DATA_PROCESSED_DIR):
    splits = load_split_patients(processed_dir)
    train_ds = BraTSSliceDataset(processed_dir, splits["train"], augment=True)
    val_ds = BraTSSliceDataset(processed_dir, splits["val"], augment=False)
    test_ds = BraTSSliceDataset(processed_dir, splits["test"], augment=False)
    return train_ds, val_ds, test_ds
