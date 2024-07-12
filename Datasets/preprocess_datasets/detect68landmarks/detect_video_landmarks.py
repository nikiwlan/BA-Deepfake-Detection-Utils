import face_alignment
from skimage import io
import os
import numpy as np
import torch
import cv2
import argparse

def calculate_percent_value(array):
    tensor = torch.sigmoid(torch.tensor(array))
    percent_value = float(tensor[0]) * 100
    return percent_value

def process_videos(fa, base_dir, dataset_name, subdataset_name):
    input_dir = os.path.join(base_dir, dataset_name, subdataset_name, "c23", "videos")
    output_dir = os.path.join(base_dir, dataset_name, subdataset_name, "c23", "landmarks")

    print(f"Processing videos in directory: {input_dir}")
    print(f"Saving landmarks to directory: {output_dir}")

    values_to_calculate = [1, 2, 3, 4, 6, 8, 10, 15]

    for value in values_to_calculate:
        percent_value = calculate_percent_value(np.array([value]))
        print(f"Für den Wert {value} ist der prozentuale Wert: {percent_value:.2f}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            if not file_name.endswith('.mp4'):
                continue

            video_path = os.path.join(root, file_name)
            print(f"Processing video: {video_path}")
            cap = cv2.VideoCapture(video_path)

            landmarks_video = []

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    print("Keine weiteren Frames im Video vorhanden.")
                    break

                print(f"Verarbeite Frame im Video: {video_path}")

                preds = fa.get_landmarks(frame)

                if preds is not None:
                    landmarks_list = preds[0].tolist()  # Use only the first detected face
                    landmarks_video.append(landmarks_list)

            cap.release()

            if landmarks_video:
                landmarks_array = np.array(landmarks_video)
                output_file_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.npy')
                np.save(output_file_path, landmarks_array)
                print(f"Saved landmarks to: {output_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Process videos for face landmarks.")
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing the video folders.')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (e.g., Forensics)')
    parser.add_argument('--subdataset_name', type=str, required=True, help='Subdataset name (e.g., RealFF)')
    args = parser.parse_args()

    # Initialize the face alignment model
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, face_detector='blazeface')

    # Process the given subdataset
    process_videos(fa, args.base_dir, args.dataset_name, args.subdataset_name)

if __name__ == "__main__":
    main()
