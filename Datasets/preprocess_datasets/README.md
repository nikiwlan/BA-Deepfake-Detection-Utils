# Preprocess Datasets

This directory contains scripts and instructions for preprocessing datasets for the RealForensics project.

## Steps to Preprocess Datasets

1. **Download the datasets (Celeb-DF or FF++)**:

    - [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics) (reduced to 500 real and 500 fake videos) 
    - [FaceForensics++](https://github.com/ondyari/FaceForensics) (reduced to 600 real and 600 fake videos)


2. **Install a necessary dependency for the script**:
    ```sh    
    pip install opencv-python
    ```

3. **Extract the video frames (For instance for FF++)**:
    ```sh    
    python3 extract_compressed_videos_FaceForensics.py --data_path [yourLocalPath]/RealForensics/data/Forensics --dataset all --compression c23
    ```
    - `--dataset`: Specify which part of FF++ (e.g., different generation methods, YouTube real, or actors real).
    - `--compression`: The compression rate of the videos (c0 = no compression, c23 = medium compression, c40 = high compression).
    - `--data_path`: Path to the dataset.


4. **Download an Alogirthm to detect 68 Lanmakarks**:
    I decided for the usage of ..

## Notes

- **Dataset Reduction**:
  
    - **Celeb-DF**: Is reduced to 500 real and 500 fake videos.
    - **FF++**: Is reduced to 600 real and 600 fake videos. Moreover, the generation methods are limited to Deepfakes, Face2Face, FaceSwap, and NeuralTextures.


## Directory Structure

- `extract_frames/`: Contains scripts for extracting single frames from videos for each dataset.
