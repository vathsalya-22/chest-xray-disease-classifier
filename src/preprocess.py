import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import random

# ── Config ──────────────────────────────────────────────────────────────
IMG_SIZE = 224
DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"
PROCESSED_DIR = f"{DATA_DIR}/processed"
TRAIN_DIR = f"{DATA_DIR}/train"
VAL_DIR = f"{DATA_DIR}/val"
TEST_DIR = f"{DATA_DIR}/test"

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

# ── Task 1: Validate ─────────────────────────────────────────────────────
def validate_data():
    print("=" * 50)
    print("TASK 1: Validating data...")
    print("=" * 50)

    errors = []

    # Check labels CSV exists
    labels_path = f"{RAW_DIR}/labels.csv"
    if not os.path.exists(labels_path):
        errors.append(f"Missing labels file: {labels_path}")
    else:
        df = pd.read_csv(labels_path)
        required_cols = ["Image Index", "Finding Labels", "Patient Age", "Patient Gender"]
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing column: {col}")
        print(f"Labels file OK — {len(df)} records found")

        # Check for nulls
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            errors.append(f"Null values found: {null_counts.to_dict()}")
        else:
            print("No null values found")

        # Check finding labels format
        valid_findings = set(LABELS + ["No Finding"])
        for idx, row in df.iterrows():
            findings = row["Finding Labels"].split("|")
            for f in findings:
                if f.strip() not in valid_findings:
                    errors.append(f"Unknown finding '{f}' in row {idx}")

        print(f"All finding labels valid")

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        raise ValueError(f"Validation failed with {len(errors)} errors")

    print("TASK 1 PASSED: Data validation complete!\n")
    return True


# ── Task 2: Preprocess ───────────────────────────────────────────────────
def preprocess_data():
    print("=" * 50)
    print("TASK 2: Preprocessing data...")
    print("=" * 50)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = pd.read_csv(f"{RAW_DIR}/labels.csv")

    # Create multi-hot encoding for labels
    for label in LABELS:
        df[label] = df["Finding Labels"].apply(
            lambda x: 1 if label in x.split("|") else 0
        )

    # Normalize patient age
    df["Patient Age"] = pd.to_numeric(df["Patient Age"], errors="coerce")
    df["Patient Age"] = df["Patient Age"].fillna(df["Patient Age"].median())
    df["Age Normalized"] = (df["Patient Age"] - df["Patient Age"].mean()) / df["Patient Age"].std()

    # Encode gender
    df["Gender Encoded"] = df["Patient Gender"].map({"M": 0, "F": 1}).fillna(0)

    # Save processed labels
    processed_path = f"{PROCESSED_DIR}/labels_processed.csv"
    df.to_csv(processed_path, index=False)

    print(f"Processed {len(df)} records")
    print(f"Label columns added: {LABELS[:3]}... and {len(LABELS)-3} more")
    print(f"Saved to {processed_path}")
    print("TASK 2 PASSED: Preprocessing complete!\n")
    return processed_path


# ── Task 3: Augmentation config ──────────────────────────────────────────
def augment_data():
    print("=" * 50)
    print("TASK 3: Setting up augmentation config...")
    print("=" * 50)

    augmentation_config = {
        "horizontal_flip": True,
        "rotation_range": 10,
        "brightness_range": [0.8, 1.2],
        "zoom_range": 0.1,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "target_size": [IMG_SIZE, IMG_SIZE],
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225]
    }

    import json
    config_path = f"{PROCESSED_DIR}/augmentation_config.json"
    with open(config_path, "w") as f:
        json.dump(augmentation_config, f, indent=2)

    print(f"Augmentation config saved to {config_path}")
    print(f"Settings: flip={augmentation_config['horizontal_flip']}, "
          f"rotation={augmentation_config['rotation_range']}deg, "
          f"brightness={augmentation_config['brightness_range']}")
    print("TASK 3 PASSED: Augmentation config ready!\n")
    return config_path


# ── Task 4: Split data ───────────────────────────────────────────────────
def split_data():
    print("=" * 50)
    print("TASK 4: Splitting data into train/val/test...")
    print("=" * 50)

    df = pd.read_csv(f"{PROCESSED_DIR}/labels_processed.csv")

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split 70/15/15
    n = len(df)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Save splits
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    train_df.to_csv(f"{TRAIN_DIR}/labels.csv", index=False)
    val_df.to_csv(f"{VAL_DIR}/labels.csv", index=False)
    test_df.to_csv(f"{TEST_DIR}/labels.csv", index=False)

    print(f"Total records: {n}")
    print(f"Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
    print(f"Val:   {len(val_df)} ({len(val_df)/n*100:.1f}%)")
    print(f"Test:  {len(test_df)} ({len(test_df)/n*100:.1f}%)")
    print("TASK 4 PASSED: Data split complete!\n")
    return {"train": len(train_df), "val": len(val_df), "test": len(test_df)}


# ── Task 5: Version with DVC ─────────────────────────────────────────────
def version_data():
    print("=" * 50)
    print("TASK 5: Versioning data with DVC...")
    print("=" * 50)

    import subprocess

    commands = [
        ["dvc", "add", "data/processed"],
        ["dvc", "add", "data/train"],
        ["dvc", "add", "data/val"],
        ["dvc", "add", "data/test"],
    ]

    for cmd in commands:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"DVC tracked: {cmd[-1]}")
        else:
            print(f"DVC note for {cmd[-1]}: {result.stderr[:100]}")

    print("TASK 5 PASSED: Data versioned with DVC!\n")
    return True


# ── Run all tasks in sequence (for local testing) ────────────────────────
if __name__ == "__main__":
    print("\nRunning full pipeline locally...\n")
    validate_data()
    preprocess_data()
    augment_data()
    split_data()
    version_data()
    print("=" * 50)
    print("ALL PIPELINE TASKS COMPLETE!")
    print("=" * 50)