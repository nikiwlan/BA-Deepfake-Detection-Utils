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

**Celeb-DF**

1. **Extract the video frames (TODO: only for Celeb-DF)**:
    ```sh    
    python3 extract_compressed_videos_FaceForensics.py --data_path [yourLocalPath]/RealForensics/data/Forensics --dataset all --compression c23
    ```
    - `--dataset`: Specify which part of FF++ (e.g., different generation methods, YouTube real, or actors real).
    - `--compression`: The compression rate of the videos (c0 = no compression, c23 = medium compression, c40 = high compression).
    - `--data_path`: Path to the dataset.


2. **Detect 68 Landmarks of the single Frames**:
    I decided for the usage of ..

**FF++**

1. **Detect 68 Landmarks of the Videos**:
    I decided for the usage of [face-alignement](https://github.com/1adrianb/face-alignment) to detect landmarks of videos I added a additional script `face-alignment_for_68_landmarks/` and save all the video in one .npy file as required.

2. **Execute the script to cropp the mouth region**
   

## Notes

- **Dataset Reduction**:
  
    - **Celeb-DF**: Is reduced to 500 real and 500 fake videos.
    - **FF++**: Is reduced to 600 real and 600 fake videos. Moreover, the generation methods are limited to Deepfakes, Face2Face, FaceSwap, and NeuralTextures. The original videos from actors and YouTube were combined into a single category called 'Real'.


## Directory Structure

- `extract_frames/`: Contains scripts for extracting single frames from videos for each dataset.
