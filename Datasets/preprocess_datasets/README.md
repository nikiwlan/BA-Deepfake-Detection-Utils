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

    
### RealForensics

1. **Detect 68 Landmarks of the Videos**:
    I decided to use [face-alignement](https://github.com/1adrianb/face-alignment) to detect landmarks of videos. I added an additional script `detect68landmarks/detect_videos` to save all the video landmarks in one .npy file as required.

2. **Execute the script to crop the faces**
   The script was not modified and is available in the RealForensics repository under `preprocessing/extract_faces.py`.


4. **Directory structure for landmarks, videos, and cropped faces:** 

**Frames:** 
data/{dataset_name}/{dataset_subset_name}/{compression}/videos/
  - 0000.png
  - 0001.png
  - ...
  - 0102.png

**Landmarken:**  
data/{dataset_name}/{dataset_subset_name}/{compression}/landmarks/
  - 0000.npy
  - 0001.npy
  - ...
  - 0102.npy

**Cropped Faces:**  
data/{dataset_name}/{dataset_subset_name}/{compression}/cropped_faces/
  - 0000.png
  - 0001.png
  - ...
  - 0102.png

**CSV files with the labels:**  
data/{dataset_name}/csv_files

- `--dataset_name`: CelebDF or FF++
- `--dataset_subset_name`: CelebDF: real or fake. FF++: fake (Deepfakes, Face2Face, etc. ) or real

**Example:**
- `data/FF++/Deepfakes/c23/videos`
- `data/FF++/Deepfakes/c23/landmarks`
- `data/FF++/Deepfakes/c23/cropped_faces`
  
  
### LipForensics

1. **Extract Frames from Videos**:
    Use the script `extract_frames.py` available in the LipForensics repository to extract frames from the videos. This script will save each frame as a .png file.

2. **Detect 68 Landmarks of the Frames**:
    After extracting the frames, use [face-alignment](https://github.com/1adrianb/face-alignment) to detect the landmarks for each frame. An additional script, `detect68landmarks/detect_videos`, can be used to save all the frame landmarks in .npy files.

3. **Execute the Script to Crop the Mouths**:
    Use the script `preprocessing/extract_mouths.py` available in the LipForensics repository. This script will crop the mouth regions from the frames based on the detected landmarks.

4. **Rename Directories and Files**:
    After processing, you need to rename the directories and files to maintain a consistent structure. Use the script `./additional_scripts/rename_directories_and_files.py` to rename directories and files as required.

5. **Directory Structure for Landmarks, Frames, and Cropped Mouths**:

    **Frames:**  
    `{dataset_name}/c23/images/`
    - 0000.png
    - 0001.png
    - ...
    - 0102.png

    **Landmarks:**  
    `{dataset_name}/c23/landmarks/`
    - 0000.npy
    - 0001.npy
    - ...
    - 0102.npy

    **Cropped Mouths:**  
    `{dataset_name}/c23/cropped_mouths/`
    - 0000.png
    - 0001.png
    - ...
    - 0102.png

**Example Dataset Structure:**
- `{dataset_name}`: FF++: Deepfakes, Face2Face, etc. | Celeb-DF: FakeCelebDF and RealCelebDF 
- `{type}`: landmarks, images, or cropped_mouths

**Example:**
- `Deepfakes/c23/images/`
- `Deepfakes/c23/landmarks/`
- `Deepfakes/c23/cropped_mouths/`


## Directory Structure

- `extract_frames/`: Contains scripts for extracting individual frames from videos for each dataset.
- `detect68landmarks/`: Contains supplementary scripts for detecting 68 landmarks within single frames and entire videos.
- `additional_scripts/`: Contains supplementary scripts, such as those for adjusting directory and file names.
- `FF++/`: Contains the download script for FF++ with the corresponding command.

