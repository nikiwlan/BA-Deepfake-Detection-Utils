import os
import sys
import torch
from torch.utils.model_zoo import load_url
import argparse
import numpy as np
from scipy.special import expit

sys.path.append('../../../')

from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet, weights
from isplutils import utils

# Set parameters
net_model = 'EfficientNetAutoAttB4'
train_db = 'DFDC'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
face_policy = 'scale'
face_size = 224
frames_per_video = 32

# Initialize network
model_url = weights.weight_url['{0}_{1}'.format(net_model, train_db)]
net = getattr(fornet, net_model)().eval().to(device)
net.load_state_dict(load_url(model_url, map_location=device, check_hash=True))

# Transformer settings
transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)

# Initialize BlazeFace and VideoReader
facedet = BlazeFace().to(device)
facedet.load_weights("../../../blazeface/blazeface.pth")
facedet.load_anchors("../../../blazeface/anchors.npy")
videoreader = VideoReader(verbose=False)

# Define video read function
video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)

# Initialize FaceExtractor
face_extractor = FaceExtractor(video_read_fn, facedet=facedet)

# Argument parser for dataset selection
parser = argparse.ArgumentParser(description="Process videos for face extraction and prediction.")
parser.add_argument('--dataset', type=str, required=True, help='Dataset to process: FF++ or Celeb-DF')
args = parser.parse_args()

if args.dataset == 'FF++':
    fake_dirs = [
        '/media/niklas/T7/FaceForensics/data/manipulated_sequences/Deepfakes',
        '/media/niklas/T7/FaceForensics/data/manipulated_sequences/Face2Face',
        '/media/niklas/T7/FaceForensics/data/manipulated_sequences/FaceShifter',
        '/media/niklas/T7/FaceForensics/data/manipulated_sequences/FaceSwap'
    ]
    prediction_file_fake = 'prediction_fake_values_ff++.txt'
elif args.dataset == 'Celeb-DF':
    fake_dirs = [
        '/media/niklas/T7/CelebDF/Celeb-synthesis'
    ]
    prediction_file_fake = 'prediction_fake_values_celeb-df.txt'
else:
    raise ValueError("Invalid dataset specified. Choose either 'FF++' or 'Celeb-DF'.")

# Helper function to get video paths from directories
def get_video_paths(directories):
    video_paths = []
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.mp4'):
                    video_paths.append(os.path.join(root, file))
    return video_paths

fake_video_paths = get_video_paths(fake_dirs)
fake_faces_results = []

for video_path in fake_video_paths:
    print(f"path: {video_path}")
    vid_fake_faces = face_extractor.process_video(video_path)
    fake_faces_results.append(vid_fake_faces)
    print("wieso")

    faces_pred_fake_all = []
    for idx, vid_faces in enumerate(fake_faces_results):
        valid_faces = [transf(image=frame['faces'][0])['image'] for frame in vid_faces if 'faces' in frame and len(frame['faces']) > 0 and isinstance(frame['faces'][0], np.ndarray)]
        print(f"Video {idx}: Anzahl der gültigen Gesichter = {len(valid_faces)}")  # Debugging-Informationen hinzufügen
        if valid_faces:
            faces_t = torch.stack(valid_faces)
            with torch.no_grad():
                faces_pred_fake = net(faces_t.to(device)).cpu().numpy().flatten()
            print(f"Prediction scores for fake video: {faces_pred_fake}")
            faces_pred_fake_all.append(faces_pred_fake)

    average_fake_scores = [np.mean(scores) for scores in faces_pred_fake_all]

    with open(prediction_file_fake, 'a') as f:
        f.write(f'Average score for FAKE video: {average_fake_scores}\n')

    faces_pred_fake = []
    faces_pred_fake_all = []
    vid_fake_faces = []
    fake_faces_results = []
    faces_t = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions for fake videos.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to process: FF++ or Celeb-DF')
    args = parser.parse_args()

    main(args.dataset)
