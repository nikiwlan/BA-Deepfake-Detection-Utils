# icpr2020dfdc

This directory contains supplementary scripts and results for the evaluation of the icpr2020dfdc model. 
The results presented in my bachelor thesis can be recreated or tested on additional datasets using the provided scripts.

## Setup

1. **Create a virtual environment**:
    ```sh
    python -m venv myenv
    ```

2. **Activate the virtual environment**:
    - On Linux:
        ```sh
        source myenv/bin/activate
        ```

3. **Install necessary packages**:
    ```sh
    pip install torch numpy scipy argparse
    ```

4. **Clone the repository**:
    ```sh
    git clone https://github.com/polimi-ispl/icpr2020dfdc
    ```

5. **Install additional dependencies**:
    ```sh
    pip install efficientnet-pytorch
    pip install -U git+https://github.com/albu/albumentations > /dev/null
    ```

6. **Navigate to the notebook directory**:
    ```sh
    cd icpr2020dfdc/notebook
    ```

7. **Place your scripts**:
    Copy all the supplementary scripts into the `notebook` root directory.

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
    - For fake videos:
        ```sh
        python process_data/calculate_fake_pred_values.py --dataset FF++
        ```
        - `--dataset`: Specifies the dataset to process. Use `FF++` or `Celeb-DF`.

    - For real videos:
        ```sh
        python process_data/calculate_real_pred_values.py --dataset FF++
        ```
        - `--dataset`: Specifies the dataset to process. Use `FF++` or `Celeb-DF`.

2. **Process Prediction Values and Compute Metrics**:
    ```sh
    python results/process_prediction_values_and_compute_metrics.py --predictions_dir /path/to/prediction_values --output_dir /path/to/metrics
    ```
    - `--predictions_dir`: Path to the directory containing the prediction value files.
    - `--output_dir`: Path to the directory where the computed metrics will be saved.

3. **View Results**:
    - Evaluation metrics can be found in `results/metrics/results.txt`.
    - ROC curve plots can be found in `results/roc_plots/`.



1. **Calculate Prediction Values**:
    - For fake videos:
        ```sh
        python process_data/calculate_fake_pred_values.py --input_dir /path/to/fake/videos --output_file /path/to/output/prediction_fake_values.txt
        ```
        - `--input_dir`: Path to the directory containing the fake video files.
        - `--output_file`: Path to the file where the prediction values will be saved.

    - For real videos:
        ```sh
        python process_data/calculate_real_pred_values.py --input_dir /path/to/real/videos --output_file /path/to/output/prediction_real_values.txt
        ```
        - `--input_dir`: Path to the directory containing the real video files.
        - `--output_file`: Path to the file where the prediction values will be saved.

2. **Process Prediction Values and Compute Metrics**:
    ```sh
    python results/process_prediction_values_and_compute_metrics.py --predictions_dir /path/to/prediction_values --output_dir /path/to/metrics
    ```
    - `--predictions_dir`: Path to the directory containing the prediction value files.
    - `--output_dir`: Path to the directory where the computed metrics will be saved.

3. **View Results**:
    - Evaluation metrics can be found in `results/metrics/results.txt`.
    - ROC curve plots can be found in `results/roc_plots/`.
