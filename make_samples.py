"""Build the 3 bundled demo cases shipped in samples/ for the Streamlit app's
"Try a sample case" mode. Picks patients from the test split (never seen
during training) and saves their normalized 4-channel volume + geometry as a
compact .npz, so the app can run inference without needing any NIfTI files
or a data download.
"""
import argparse
import json
import os

import numpy as np

from constants import DATA_PROCESSED_DIR, DATA_RAW_DIR, SAMPLES_DIR
from preprocess import discover_patients, load_patient_volumes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", default=DATA_RAW_DIR)
    parser.add_argument("--processed_dir", default=DATA_PROCESSED_DIR)
    parser.add_argument("--samples_dir", default=SAMPLES_DIR)
    parser.add_argument("--patient_ids", nargs="*", default=None,
                         help="Specific patient IDs to bundle; defaults to the "
                              "first 3 patients in the test split.")
    args = parser.parse_args()

    os.makedirs(args.samples_dir, exist_ok=True)
    with open(os.path.join(args.processed_dir, "split_patients.json")) as f:
        splits = json.load(f)

    patient_ids = args.patient_ids or splits["test"][:3]
    all_patients = discover_patients(args.raw_dir)

    for patient_id in patient_ids:
        patient_dir = all_patients[patient_id]
        image, _, zooms, affine = load_patient_volumes(patient_dir, load_seg=False)
        out_path = os.path.join(args.samples_dir, f"{patient_id}.npz")
        np.savez_compressed(out_path, image=image.astype(np.float32),
                             zooms=np.array(zooms, dtype=np.float32),
                             affine=affine.astype(np.float64))
        print(f"[make_samples] wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
