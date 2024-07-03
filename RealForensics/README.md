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

## Download and preprocess the datasets

1. **Download the datasets (Celeb-DF or FF++)**:

    - [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics) (reduced to 500 real and 500 fake videos) 
    - [FaceForensics++](https://github.com/ondyari/FaceForensics) (reduced to 600 real and 600 fake videos)

2. **Install a necessary dependency for the script**:
    ```sh    
    pip install opencv-python
    ```

3. **Extract the single frames of the videos with the script within the dataset (e.g., FF++)**:
    ```sh    
    python3 extract_compressed_videos_FaceForensics.py --data_path [yourLocalPath]/RealForensics/data/Forensics --dataset all --compression c23
    ```
    - `--dataset`: Specify which part of FF++ (e.g., different generation methods, YouTube real, or actors real).
    - `--compression`: The compression rate of the videos (c0 = no compression, c23 = medium compression, c40 = high compression).
    - `--data_path`: Path to the dataset.

2. **Download an Alogirthm to detect 68 Lanmakarks**:
    I decided for the usage of ... 
   

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

## Usage

1. **Calculate Prediction Values**:
   
    - **Note: Adjust the paths of the datasets in the scripts!**
      
    - For fake videos:
        ```sh
        python3 process_data/calculate_fake_pred_values.py --dataset FF++
        ```
        - `--dataset`: Specifies the dataset to process. Use `FF++` or `Celeb-DF`.

    - For real videos:
        ```sh
        python3 process_data/calculate_real_pred_values.py --dataset FF++
        ```
        - `--dataset`: Specifies the dataset to process. Use `FF++` or `Celeb-DF`.
          
    - **Note: The newly generated prediction values are created in the icpr2020dfdc root directory!**

2. **Process Prediction Values and Compute Metrics**:
    ```sh
    python3 results/process_prediction_values_and_compute_metrics.py --threshold [value] --propThreshold [value] --real_path [path_to_real_predictions] --fake_path [path_to_fake_predictions]
    ```
    - `--threshold`: Threshold value.
    - `--propThreshold`: Probability threshold value.
    - `--real_path`: Path to the .txt file with real video predictions.
    - `--fake_path`: Path to the .txt file with fake video predictions.

   **Example Command**
   ```sh
   python3 results/process_prediction_values_and_compute_metrics.py --threshold 0 --propThreshold 0 --real_path prediction_real_values_ff++.txt --fake_path prediction_fake_values_ff++.txt
   ```

3. **View Results**:
    - Evaluation metrics can be found in `results/metrics/results.txt`.
    - ROC curve plots can be found in `results/roc_plots/`.
