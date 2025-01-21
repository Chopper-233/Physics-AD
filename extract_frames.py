import argparse
import os
from pathlib import Path
from tqdm import tqdm
import cv2


def extract_frames(video_path, frames_dir):
    video_name = Path(video_path).stem

    video_frames_dir = frames_dir
    os.makedirs(video_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(video_frames_dir, f"frame_{frame_count:04d}.png")

        cv2.imwrite(frame_path, frame)

        frame_count += 1

    cap.release()
    # print(f"Extracted {frame_count} frames from {video_path} to {video_frames_dir}")
    return video_name, frame_count


def main(videos_dir, target_path, task):
    for obj in os.listdir(videos_dir):
        # if obj!="ball":
            # print("what?")
        os.makedirs(os.path.join(target_path, obj), exist_ok=True)
        id = 0
        if task=='train':
            frames_path = os.path.join(target_path, obj, 'train', 'frames')
            os.makedirs(frames_path, exist_ok=True)
            for vid in tqdm(os.listdir(os.path.join(os.path.join(videos_dir, obj, 'train')))):
                extract_frames(os.path.join(os.path.join(videos_dir, obj, 'train', vid)), os.path.join(frames_path, f"{id:04d}_{obj}_train"))
                id+=1
        else:
            frames_path = os.path.join(target_path, obj, 'test', 'frames')
            os.makedirs(frames_path, exist_ok=True)
            for abn in os.listdir(os.path.join(os.path.join(videos_dir, obj, 'test'))):
                for vid in tqdm(os.listdir(os.path.join(os.path.join(videos_dir, obj, 'test', abn)))):
                    extract_frames(os.path.join(os.path.join(videos_dir, obj, 'test', abn, vid)), os.path.join(frames_path, f"{id:04d}_{obj}_{abn}"))
                    id+=1
        print(f"{obj} finished!")



if __name__ == "__main__":
    video_path = "/home/dataset/gy/dynamic"
    target_path = "/home/dataset/gy/phys/"
    os.makedirs(target_path, exist_ok=True)
    for task in ['train','test']:
        main(video_path, target_path, task)
        print(f"{task} finished!")
    # main(video_path, frame_path, anno_path)
