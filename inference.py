"""Full-volume inference: NIfTI directory in -> 3D predicted mask + tumor
volumes out. Also the shared prediction path used by evaluate.py and app.py
so all three stay numerically consistent."""
import numpy as np
import torch

from constants import BEST_CHECKPOINT_PATH, CROP_SIZE, REGION_LABELS
from model import build_model, region_mask
from preprocess import center_crop_or_pad_2d, load_patient_volumes, uncrop_or_pad_2d


def load_model(checkpoint_path=BEST_CHECKPOINT_PATH, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


@torch.no_grad()
def predict_cropped_volume(model, image, device, batch_size=16):
    """image: (4, size, size, D) float32 array, already center-cropped/padded.
    Returns predicted label volume (size, size, D) as uint8."""
    num_slices = image.shape[-1]
    preds = np.zeros((image.shape[1], image.shape[2], num_slices), dtype=np.uint8)

    for start in range(0, num_slices, batch_size):
        end = min(start + batch_size, num_slices)
        batch = image[:, :, :, start:end]                       # (4, H, W, b)
        batch = np.transpose(batch, (3, 0, 1, 2))                # (b, 4, H, W)
        batch_t = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to(device)
        logits = model(batch_t)
        batch_preds = torch.argmax(logits, dim=1).cpu().numpy()  # (b, H, W)
        preds[:, :, start:end] = np.transpose(batch_preds, (1, 2, 0)).astype(np.uint8)

    return preds


def segment_volume(patient_dir, checkpoint_path=BEST_CHECKPOINT_PATH, model=None, device=None):
    """Run the full preprocessing + per-slice prediction + reassembly pipeline
    on a directory containing the 4 modality NIfTI volumes.

    Returns:
        pred_mask: uint8 array, same (H, W, D) shape as the input volumes,
                   remapped label convention {0,1,2,3}.
        zooms: voxel spacing (mm) from the NIfTI header.
        affine: reference affine, for re-saving the prediction as NIfTI.
    """
    if model is None:
        model, device = load_model(checkpoint_path, device)
    device = device or next(model.parameters()).device

    image, _, zooms, affine = load_patient_volumes(patient_dir, load_seg=False)
    orig_h, orig_w, num_slices = image.shape[1], image.shape[2], image.shape[3]

    cropped = np.stack([
        np.stack([center_crop_or_pad_2d(image[c, :, :, d], size=CROP_SIZE)
                  for c in range(image.shape[0])], axis=0)
        for d in range(num_slices)
    ], axis=-1)  # (4, CROP_SIZE, CROP_SIZE, D)

    cropped_pred = predict_cropped_volume(model, cropped, device)  # (CROP_SIZE, CROP_SIZE, D)

    pred_mask = np.stack([
        uncrop_or_pad_2d(cropped_pred[:, :, d], orig_h, orig_w, size=CROP_SIZE)
        for d in range(num_slices)
    ], axis=-1).astype(np.uint8)

    return pred_mask, zooms, affine


def compute_region_volumes_cm3(pred_mask, zooms):
    """voxel volume (mm^3) * voxel count / 1000 -> cm^3, per BraTS region."""
    voxel_vol_mm3 = float(np.prod(zooms))
    volumes = {}
    for region in REGION_LABELS:
        mask = region_mask(torch.from_numpy(pred_mask.astype(np.int64)), region).numpy()
        volumes[region] = mask.sum() * voxel_vol_mm3 / 1000.0
    return volumes


def save_prediction_nifti(pred_mask, affine, out_path):
    import nibabel as nib
    nib.save(nib.Nifti1Image(pred_mask.astype(np.uint8), affine), out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patient_dir")
    parser.add_argument("--checkpoint", default=BEST_CHECKPOINT_PATH)
    parser.add_argument("--out", default="prediction.nii.gz")
    args = parser.parse_args()

    pred_mask, zooms, affine = segment_volume(args.patient_dir, args.checkpoint)
    save_prediction_nifti(pred_mask, affine, args.out)
    volumes = compute_region_volumes_cm3(pred_mask, zooms)
    print(f"Saved prediction to {args.out}")
    for region, cm3 in volumes.items():
        print(f"  {region}: {cm3:.2f} cm^3")
