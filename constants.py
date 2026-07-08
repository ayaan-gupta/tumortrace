"""Shared constants for the TumorTrace pipeline.

Every module (preprocess, dataset, model, train, evaluate, inference, app)
imports from here so that label conventions, geometry, and class metadata
stay in exactly one place.
"""

# --- Label remapping -------------------------------------------------------
# Raw BraTS labels: 0=background, 1=NCR/NET, 2=edema, 4=enhancing tumor.
# Label 4 is remapped to 3 so the 4-way softmax head sees contiguous
# class indices {0, 1, 2, 3}.
LABEL_MAP = {0: 0, 1: 1, 2: 2, 4: 3}

NUM_CLASSES = 4
CLASS_NAMES = {
    0: "background",
    1: "necrotic_non_enhancing_core",
    2: "edema",
    3: "enhancing_tumor",
}

# --- BraTS evaluation regions (defined over the *remapped* labels) --------
# Whole Tumor = union of all tumor labels.
# Tumor Core = necrotic core + enhancing tumor (excludes edema).
# Enhancing Tumor = enhancing tumor only.
REGION_LABELS = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "ET": (3,),
}

# --- Modalities --------------------------------------------------------
MODALITIES = ("t1", "t1ce", "t2", "flair")
NUM_MODALITIES = len(MODALITIES)

# --- Geometry ------------------------------------------------------------
RAW_SLICE_SIZE = 240      # native BraTS axial in-plane size
CROP_SIZE = 192           # center-crop/pad target used throughout training/inference
NUM_SLICES = 155          # native BraTS axial slice count (3rd axis)

# --- Sampling --------------------------------------------------------
HARD_NEGATIVE_FRACTION = 0.10  # fraction of tumor-free slices kept per patient

# --- Splits --------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_STATE = 42

# --- Training defaults (see README §6 for rationale) --------------------
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 7
LR_PLATEAU_FACTOR = 0.5
LR_PLATEAU_PATIENCE = 4

# --- Paths --------------------------------------------------------
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
CHECKPOINT_DIR = "checkpoints"
BEST_CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/best_model.pt"
RESULTS_DIR = "results"
SAMPLES_DIR = "samples"

# --- Overlay colors (RGB, 0-1 floats) used by app.py / evaluate.py -------
# red = enhancing tumor, yellow = edema, blue = necrotic/non-enhancing core
OVERLAY_COLORS = {
    1: (0.0, 0.0, 1.0),   # necrotic/non-enhancing core -> blue
    2: (1.0, 1.0, 0.0),   # edema -> yellow
    3: (1.0, 0.0, 0.0),   # enhancing tumor -> red
}
OVERLAY_ALPHA = 0.4

DISCLAIMER = (
    "**This is a research and educational prototype, not a medical device.** "
    "It has not been clinically validated and must never be used for actual "
    "diagnosis or treatment decisions. Segmentation outputs should only ever "
    "be interpreted by a qualified radiologist or neuro-oncologist."
)
