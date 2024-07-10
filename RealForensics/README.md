# RealForensics

This directory contains supplementary scripts and results for the evaluation of the icpr2020dfdc model. 
The results presented in my bachelor thesis can be recreated or tested on additional datasets using the provided scripts.

## Setup

1. **Clone the RealForensics git repository**:
    ```sh
    git clone https://github.com/ahaliassos/RealForensics.git
    ```

2. **Navigate to the repository**:
    ```sh
    cd RealForensics
    ```

3. **Download Miniconda**:
    ```sh
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh
    ```

4. **Run the installer**:
    ```sh
    bash Miniconda3-latest-Linux-x86_64.sh
    ```

5. **Initialize Conda**:
    ```sh
    source ~/miniconda3/bin/activate
    conda init
    ```

6. **Create the environment by running the environment.yml file**:
    ```sh
    conda env create -f environment.yml
    ```

7. **Activate the newly created environment**:
    ```sh
    conda activate el_dorado
    ```

## Dataset Preprocessing

The steps to preprocess the datasets are located in a different directory. Please follow the instructions provided in the following location:

[`Preprocess Datasets`](../Datasets/preprocess_datasets)

After extracting the frames in the videos and detecting the landmarks the folders have to be placed as following:

## Video Placement

Place the videos in the corresponding directories. For example, for datasets like FaceForensics++, FaceShifter, and DeeperForensics, use the following structure:

`data/Forensics/{type}/c23/videos`

Here, `{type}` can be one of the following:
- Real
- Deepfakes
- FaceSwap
- Face2Face
- NeuralTextures

**Example** for FaceForensics++ Deepfakes:
`data/Forensics/Deepfakes/c23/videos`

## Facial Landmarks Placement

Place the facial landmarks in the corresponding folders as `.npy` files. Use the same directory structure as for the videos, but replace `videos` with `landmarks`. Each landmark file should have the same name as its corresponding video, except that it ends in `.npy`.

**Example** for FaceForensics++ Deepfakes landmarks:
`data/Forensics/Deepfakes/c23/landmarks`

In my case, the 68 landmarks are computed with face alignment.

## Cross-Dataset Evaluation 
    ```sh
    pip install hydra-core pytorchvideo
    ```



## Directory Structure

- `process_data/`: Contains scripts for processing labeled .mp4 folders. There is also a sub-folder with predicted values for the FF++ and Celeb-DF datasets.
  - `calculate_fake_pred_values.py`: Script to calculate prediction values for fake videos.
  - `calculate_real_pred_values.py`: Script to calculate prediction values for real videos.
  - `prediction_values/`: Contains real and fake prediction values for labeled datasets: FF++ and Celeb-DF.
    - `prediction_fake_values_celeb-df.txt`: Prediction values for 500 fake videos from the Celeb-DF dataset.
    - `prediction_fake_values_ff++`: Prediction values for 600 fake videos from the FaceForensics++ dataset.
    - `prediction_real_values_celeb-df.txt`: Prediction values for 500 real videos from the Celeb-DF dataset.
    - `prediction_real_values_ff++`: Prediction values for 600 real videos from the FaceForensics++ dataset.

- `results/`: Contains results and metrics from the evaluation.
  - `process_prediction_values_and_compute_metrics.py`: Script to process prediction values and compute evaluation metrics.
  - `metrics/`: Contains detailed metrics for the evaluated models.
    - `results.txt`: Text file with various metrics and details on different thresholds and probability thresholds.
  - `roc_plots/`: Contains ROC curve plots on various thresholds and probability thresholds.
    - Various ROC curve plots as .png files.

TODO : ## Usage

1. **Pretrained Model**
   - Download the [`pretrained_model`](https://drive.google.com/file/d/1nqEVlRN51WyzMWSeB7x9okcaegFgA-BQ/view) on google drive

2. **Execute the Script to Compute Metrics**:
   
    python stage2/eval.py model.weights_filename=realforensics_ff.pth
