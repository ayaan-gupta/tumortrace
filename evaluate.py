"""Patient-level (3D) evaluation on the held-out BraTS test split.

For each test patient, reconstructs the full predicted volume from per-slice
2D predictions (mirroring how inference.py works) and compares it against
ground truth in native 240x240xD space. Reports, per BraTS region (WT/TC/ET):
Dice, 95th-percentile Hausdorff Distance, sensitivity, specificity.

Also renders an 8-slice qualitative grid (GT vs. prediction) to
results/qualitative_examples.png.
"""
import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless/CI environments
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.metrics import compute_hausdorff_distance

from constants import (
    BEST_CHECKPOINT_PATH,
    CROP_SIZE,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    OVERLAY_ALPHA,
    OVERLAY_COLORS,
    REGION_LABELS,
    RESULTS_DIR,
)
from inference import load_model, predict_cropped_volume
from model import region_mask
from preprocess import center_crop_or_pad_2d, discover_patients, load_patient_volumes


def _region_confusion(pred_region, gt_region, eps=1e-8):
    pred_region = pred_region.astype(bool)
    gt_region = gt_region.astype(bool)
    tp = np.logical_and(pred_region, gt_region).sum()
    fn = np.logical_and(~pred_region, gt_region).sum()
    fp = np.logical_and(pred_region, ~gt_region).sum()
    tn = np.logical_and(~pred_region, ~gt_region).sum()
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return dice, sensitivity, specificity


def _hd95(pred_region, gt_region, spacing):
    if not gt_region.any() and not pred_region.any():
        return 0.0
    if not gt_region.any() or not pred_region.any():
        return None  # undefined (only one of pred/gt has any foreground) -> excluded from average
    pred_t = torch.from_numpy(pred_region[None, None].astype(np.float32))
    gt_t = torch.from_numpy(gt_region[None, None].astype(np.float32))
    hd = compute_hausdorff_distance(
        pred_t, gt_t, include_background=True, percentile=95, spacing=[float(s) for s in spacing]
    )
    return float(hd.item())


def evaluate_patient(patient_id, patient_dir, model, device):
    image, seg, zooms, _ = load_patient_volumes(patient_dir, load_seg=True)
    orig_h, orig_w, num_slices = image.shape[1], image.shape[2], image.shape[3]

    cropped = np.stack([
        np.stack([center_crop_or_pad_2d(image[c, :, :, d], size=CROP_SIZE)
                  for c in range(image.shape[0])], axis=0)
        for d in range(num_slices)
    ], axis=-1)
    cropped_pred = predict_cropped_volume(model, cropped, device)

    cropped_gt = np.stack([
        center_crop_or_pad_2d(seg[:, :, d], size=CROP_SIZE) for d in range(num_slices)
    ], axis=-1)

    results = {}
    for region in REGION_LABELS:
        pred_region = region_mask(torch.from_numpy(cropped_pred.astype(np.int64)), region).numpy()
        gt_region = region_mask(torch.from_numpy(cropped_gt.astype(np.int64)), region).numpy()
        dice, sens, spec = _region_confusion(pred_region, gt_region)
        hd95 = _hd95(pred_region, gt_region, zooms)
        results[region] = {"dice": dice, "sensitivity": sens, "specificity": spec, "hd95": hd95}
    return results


def run_evaluation(raw_dir, processed_dir, checkpoint_path, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(processed_dir, "split_patients.json")) as f:
        splits = json.load(f)
    test_ids = set(splits["test"])

    all_patients = discover_patients(raw_dir)
    test_patients = {pid: d for pid, d in all_patients.items() if pid in test_ids}
    if not test_patients:
        raise SystemExit(f"No test patients found in {raw_dir} matching split_patients.json")

    model, device = load_model(checkpoint_path)

    per_region_metrics = {r: {"dice": [], "sensitivity": [], "specificity": [], "hd95": []}
                          for r in REGION_LABELS}

    for patient_id, patient_dir in test_patients.items():
        patient_results = evaluate_patient(patient_id, patient_dir, model, device)
        print(f"[evaluate] {patient_id}: "
              + ", ".join(f"{r} Dice={v['dice']:.3f}" for r, v in patient_results.items()))
        for region, metrics in patient_results.items():
            per_region_metrics[region]["dice"].append(metrics["dice"])
            per_region_metrics[region]["sensitivity"].append(metrics["sensitivity"])
            per_region_metrics[region]["specificity"].append(metrics["specificity"])
            if metrics["hd95"] is not None:
                per_region_metrics[region]["hd95"].append(metrics["hd95"])

    summary = {}
    for region, metrics in per_region_metrics.items():
        summary[region] = {k: float(np.mean(v)) if v else float("nan") for k, v in metrics.items()}

    _write_metrics_table(summary, len(test_patients), os.path.join(results_dir, "metrics_table.md"))
    make_qualitative_grid(processed_dir, splits["test"], model, device,
                          os.path.join(results_dir, "qualitative_examples.png"))
    return summary


def _write_metrics_table(summary, n_patients, out_path):
    lines = [
        f"# TumorTrace — Test Set Results ({n_patients} patients)\n",
        "| Region | Dice | HD95 (mm) | Sensitivity | Specificity |",
        "|---|---|---|---|---|",
    ]
    for region in ("WT", "TC", "ET"):
        m = summary[region]
        lines.append(
            f"| {region} | {m['dice']:.3f} | {m['hd95']:.2f} | "
            f"{m['sensitivity']:.3f} | {m['specificity']:.3f} |"
        )
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[evaluate] Wrote {out_path}")


def _overlay_rgb(base_gray, label_slice):
    rgb = np.stack([base_gray] * 3, axis=-1)
    for label, color in OVERLAY_COLORS.items():
        mask = label_slice == label
        for c in range(3):
            rgb[..., c] = np.where(mask, (1 - OVERLAY_ALPHA) * rgb[..., c] + OVERLAY_ALPHA * color[c], rgb[..., c])
    return np.clip(rgb, 0, 1)


def make_qualitative_grid(processed_dir, test_patient_ids, model, device, out_path, n_examples=8, seed=42):
    from glob import glob

    candidates = []
    for pid in test_patient_ids:
        for mask_path in glob(os.path.join(processed_dir, f"mask_{pid}_*.npy")):
            mask = np.load(mask_path)
            if np.any(mask > 0):
                candidates.append(mask_path)

    rng = random.Random(seed)
    chosen = rng.sample(candidates, min(n_examples, len(candidates)))

    fig, axes = plt.subplots(len(chosen), 2, figsize=(6, 3 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]

    for row, mask_path in enumerate(chosen):
        image_path = os.path.join(os.path.dirname(mask_path),
                                    os.path.basename(mask_path).replace("mask_", "image_", 1))
        image = np.load(image_path)  # (4, H, W)
        gt_mask = np.load(mask_path)

        flair = image[3]  # flair is last in MODALITIES order
        flair_norm = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)

        with torch.no_grad():
            batch = torch.from_numpy(image[None].astype(np.float32)).to(device)
            logits = model(batch)
            pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()

        axes[row, 0].imshow(_overlay_rgb(flair_norm, gt_mask))
        axes[row, 0].set_title("Ground truth" if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(_overlay_rgb(flair_norm, pred_mask))
        axes[row, 1].set_title("Prediction" if row == 0 else "")
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", default=DATA_RAW_DIR)
    parser.add_argument("--processed_dir", default=DATA_PROCESSED_DIR)
    parser.add_argument("--checkpoint", default=BEST_CHECKPOINT_PATH)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    args = parser.parse_args()

    summary = run_evaluation(args.raw_dir, args.processed_dir, args.checkpoint, args.results_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
