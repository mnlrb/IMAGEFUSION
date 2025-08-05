import os
import pandas as pd
from pathlib import Path
import shutil

FRAMES_DIR = "D:/frames"
CSV_PATH = "D:/frame_labels.csv"
OUTPUT_CSV = "D:/frame_labels_aligned.csv"

# Create aligned directories
os.makedirs(f"{FRAMES_DIR}/RGB_aligned", exist_ok=True)
os.makedirs(f"{FRAMES_DIR}/IR_aligned", exist_ok=True)

df = pd.read_csv(CSV_PATH)
aligned_rows = []

rgb_files = sorted((Path(FRAMES_DIR) / "RGB").glob("*.jpg"))
ir_files = sorted((Path(FRAMES_DIR) / "IR").glob("*.jpg"))
max_len = min(len(rgb_files), len(ir_files))

print(f"Before alignment: RGB {len(rgb_files)} vs IR {len(ir_files)} -> using {max_len}")

for i in range(max_len):
    rgb_src = rgb_files[i]
    ir_src = ir_files[i]
    rgb_dst = Path(FRAMES_DIR) / "RGB_aligned" / rgb_src.name
    ir_dst = Path(FRAMES_DIR) / "IR_aligned" / ir_src.name

    shutil.copy(rgb_src, rgb_dst)
    shutil.copy(ir_src, ir_dst)

    frame_id = int(rgb_src.stem)
    row = df[df["frame_id"] == frame_id]
    if not row.empty:
        aligned_rows.append([frame_id, row.iloc[0]["label"], row.iloc[0]["video_id"]])

# Save new CSV
aligned_df = pd.DataFrame(aligned_rows, columns=["frame_id", "label", "video_id"])
aligned_df.to_csv(OUTPUT_CSV, index=False)

print(f"Aligned CSV saved to {OUTPUT_CSV}")
