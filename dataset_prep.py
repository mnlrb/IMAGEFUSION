import os
import csv

# Directories
RGB_DIR = r"D:/chrome download/Frame Pairs/254p RGB Images"
IR_DIR = r"D:/chrome download/Frame Pairs/254p Thermal Images"
LABEL_FILE = r"D:/chrome download/Frame Pair Labels.txt"
CSV_PATH = r"D:/frame_labels_aligned.csv"

# Load frame intervals and labels from the text file
def load_intervals(label_file):
    intervals = []
    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Skip header or invalid lines
            if not parts[0].isdigit():
                continue
            start, end, label = int(parts[0]), int(parts[1]), parts[2]
            intervals.append((start, end, label))
    return intervals

# Get fire/smoke label for a given frame_id
def get_label(frame_id, intervals):
    for start, end, label in intervals:
        if start <= frame_id <= end:
            fire = 1 if label[0] == "Y" else 0
            smoke = 1 if label[1] == "Y" else 0
            return fire, smoke
    return 0, 0

if __name__ == "__main__":
    intervals = load_intervals(LABEL_FILE)
    print(f"Loaded {len(intervals)} intervals from label file.")

    rgb_files = sorted(os.listdir(RGB_DIR))
    ir_files = sorted(os.listdir(IR_DIR))

    assert len(rgb_files) == len(ir_files), "RGB and IR frames count mismatch."

    with open(CSV_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame_id", "fire", "smoke", "rgb_path", "ir_path"])

        for idx, (rgb_file, ir_file) in enumerate(zip(rgb_files, ir_files), start=1):
            rgb_path = os.path.join(RGB_DIR, rgb_file)
            ir_path = os.path.join(IR_DIR, ir_file)
            fire, smoke = get_label(idx, intervals)
            writer.writerow([idx, fire, smoke, rgb_path, ir_path])

            if idx % 5000 == 0:
                print(f"Processed {idx} frames")

    print(f"CSV saved to {CSV_PATH}")

