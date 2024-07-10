# Preprocess Datasets

This directory contains scripts and instructions for preprocessing datasets for both the LipForensics and RealForensics models.

## Steps to Preprocess Datasets

1. **Download the datasets (Celeb-DF or FF++)**:

    - [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics) 
    - [FaceForensics++](https://github.com/ondyari/FaceForensics) 


2. **Install a necessary dependency for the script**:
    ```sh    
    pip install opencv-python
    ```
## Notes

- **Dataset Reduction**:
  
    - **Celeb-DF**: Is reduced to 500 real and 500 fake videos.
    - **FF++**: Is reduced to 600 real and 600 fake videos. Moreover, the generation methods are limited to Deepfakes, Face2Face, FaceSwap, and NeuralTextures. The original videos from actors and YouTube were combined into a single category called 'Real'.    


**LipForensics**

1. **Extract the video frames (for instace for FF++)**:
    ```sh    
    python3 extract_compressed_videos_FaceForensics.py --data_path [yourLocalPath]/RealForensics/data/Forensics --dataset all --compression c23
    ```
    - `--dataset`: Specify which part of FF++ (e.g., different generation methods, YouTube real, or actors real).
    - `--compression`: The compression rate of the videos (c0 = no compression, c23 = medium compression, c40 = high compression).
    - `--data_path`: Path to the dataset.

Benennung der Frames nach Forlaufenden nummern


2. **Detect 68 Landmarks of the single Frames**:
    I decided for the usage of [face-alignement](https://github.com/1adrianb/face-alignment) to detect the landmarks of the single frames. The necessary script is on `detect68landmarks/detect_videos`.

3. **Execute the script to cropp the mouth region**
   Das skript dafür wurde nicht angepasst und liegt im Repository des jeweiligen Models LipForensics oder RealForensics.

4. **Diese Ordnerstrukur soll die landmarks, videos und cropped faces sein** TODO!

## FF++:

**Frames:**  
data/Forensics/{dataset_name}/{compression}/videos/
  - 0000.png
  - 0001.png
  - ...
  - 0102.png

**Landmarken:**  
data/Forensics/{dataset_name}/{compression}/landmarks/
  - 0000.npy
  - 0001.npy
  - ...
  - 0102.npy

**Cropped Faces:**  
data/Forensics/{dataset_name}/{compression}/cropped_mouths/
  - 0000.png
  - 0001.png
  - ...
  - 0102.png


## Celeb-DF

**Frames:**  
data/datasets/CelebDF/{dataset_name}/images/{video}/
  - 0000.png
  - 0001.png
  - ...
  - 0102.png

**Landmarken:**  
data/datasets/CelebDF/{dataset_name}/landmarks/{video}/
  - 0000.npy
  - 0001.npy
  - ...
  - 0102.npy

    
**RealForensics**

1. **Detect 68 Landmarks of the Videos**:
    I decided for the usage of [face-alignement](https://github.com/1adrianb/face-alignment) to detect landmarks of videos I added a additional script `detect68landmarks/detect_videos` and save all the video in one .npy file as required.

2. **Execute the script to cropp the mouth region**
   Das skript dafür wurde nicht angepasst und liegt im Repository von RealForensics unter `preprocessing/extract_faces.py` .


4. **Diese Ordnerstrukur soll die landmarks, videos und cropped faces sein** 

## FF++

# Frames
data/{dataset_name}/{dataset_name}/{compression}/videos/
  - 0000.png
  - 0001.png
  - ...
  - 0102.png

# Landmarken
data/{dataset_name}/{dataset_name}/{compression}/landmarks/
  - 0000.npy
  - 0001.npy
  - ...
  - 0102.npy

# Cropped Faces 
data/{dataset_name}/{dataset_name}/{compression}/cropped_faces/
  - 0000.png
  - 0001.png
  - ...
  - 0102.png

- `--dataset_name`: CelebDF or FF++
- `--dataset_subset_name`: CelebDF: real or fake. FF++: fake (name of the generation method) or real 


## Directory Structure

- `extract_frames/`: Contains scripts for extracting single frames from videos for each dataset.
- `detect68landmarks/` Contains supplemente scripts for detect 68 landmarks within single frames and entire videos.
