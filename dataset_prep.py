import cv2
import os
import csv
from pathlib import Path

# 输入视频文件夹（RGB 和 IR 视频放在一起）
VIDEO_DIR = "D:/chrome download/All Video Pairs"
OUTPUT_DIR = "D:/frames"
CSV_PATH = "D:/frame_labels.csv"

# 每隔多少帧抽取一帧
FRAME_SAMPLE_RATE = 10

os.makedirs(f"{OUTPUT_DIR}/RGB", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/IR", exist_ok=True)

# 模拟标签逻辑，这里你应该换成真实标签
def get_label(frame_id, video_id):
    if frame_id % 100 < 20:
        return "YY"
    elif frame_id % 100 < 50:
        return "YN"
    else:
        return "NN"

with open(CSV_PATH, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["frame_id", "label", "video_id"])

    frame_counter = 0
    for video_file in sorted(os.listdir(VIDEO_DIR)):
        if not video_file.lower().endswith((".mp4", ".avi", ".mov")):
            continue

        video_path = os.path.join(VIDEO_DIR, video_file)
        cap = cv2.VideoCapture(video_path)

        if "IR" in video_file.upper():
            modality = "IR"
        else:
            modality = "RGB"

        video_id = Path(video_file).stem.replace(" ", "_")  # 视频名字作为 video_id
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % FRAME_SAMPLE_RATE == 0:
                frame_counter += 1
                fname = f"{frame_counter:06d}.jpg"
                save_path = os.path.join(OUTPUT_DIR, modality, fname)
                cv2.imwrite(save_path, frame)

                label = get_label(frame_counter, video_id)
                writer.writerow([frame_counter, label, video_id])

            frame_id += 1

        cap.release()
        print(f"{video_file} done.")

print(f"CSV saved to {CSV_PATH}")
