import os
import csv
import pandas as pd

print("Setting up data directories...")
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/sample", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/val", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

# The 14 diseases in NIH ChestX-ray14 dataset
labels = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

print(f"Dataset has {len(labels)} disease classes: {labels}")

# Create labels.csv matching the exact official NIH format
with open("data/raw/labels.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Image Index", "Finding Labels", "Follow-up #",
        "Patient ID", "Patient Age", "Patient Gender",
        "View Position", "OriginalImage Width", "OriginalImage Height"
    ])
    sample_rows = [
        ["00000001_000.png", "Cardiomegaly", 0, 1, 58, "M", "PA", 2682, 2749],
        ["00000001_001.png", "Cardiomegaly|Emphysema", 1, 1, 58, "M", "PA", 2682, 2749],
        ["00000001_002.png", "Cardiomegaly|Effusion", 2, 1, 58, "M", "PA", 2682, 2749],
        ["00000002_000.png", "No Finding", 0, 2, 81, "M", "PA", 2500, 2048],
        ["00000003_000.png", "Hernia", 0, 3, 81, "F", "AP", 2500, 2048],
        ["00000004_000.png", "Atelectasis|Effusion", 0, 4, 72, "F", "PA", 2048, 2500],
        ["00000005_000.png", "Pneumonia", 0, 5, 45, "M", "PA", 2682, 2749],
        ["00000006_000.png", "No Finding", 0, 6, 33, "F", "PA", 2048, 2048],
        ["00000007_000.png", "Nodule|Mass", 0, 7, 60, "M", "PA", 2500, 2749],
        ["00000008_000.png", "Pneumothorax", 0, 8, 29, "F", "AP", 2682, 2749],
    ]
    writer.writerows(sample_rows)

# Verify it loaded correctly
df = pd.read_csv("data/raw/labels.csv")
print(f"\nLabels CSV created successfully!")
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nSample data:")
print(df[["Image Index", "Finding Labels", "Patient Age", "Patient Gender"]].to_string())
print("\nLocal setup complete!")
print("Full 112,000 image dataset will be downloaded on Google Colab.")