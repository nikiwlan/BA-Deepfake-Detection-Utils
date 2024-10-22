# icpr2020dfdc

This directory contains supplementary scripts and results for the evaluation of the icpr2020dfdc model. 
The results presented in my bachelor thesis can be recreated or tested on additional datasets using the provided scripts.


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
   

## Setup

1. **Create a virtual environment**:
    ```sh
    python3 -m venv myenv
    ```

2. **Activate the virtual environment**:
    ```sh
    source myenv/bin/activate
    ```

3. **Install necessary packages**:
    ```sh
    pip install torch numpy scipy argparse torchvision matplotlib scikit-learn opencv-python

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
7. **Clone this repository**:
    ```sh
    git clone https://github.com/nikiwlan/BA-Deepfake-Detection-Utils.git
    ```
8. **Navigate to the icpr2020dfdc directory**:
    ```sh
    cd BA-Deepfake-Detection-Utils/icpr2020dfdc
    ```    

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
    python3 results/process_prediction_values_and_compute_metrics.py --threshold [value] --propThreshold [value] --dataset [dataset_name]
    ```
    - `--threshold`: Threshold value. This value determines the cutoff point for classifying predictions as positive or negative. For example, predictions above the threshold are classified as positive, and those below are classified as negative. Adjusting this value changes the sensitivity of the classification.
    - `--propThreshold`: Probability threshold value. This value defines a range around the main threshold to further refine classifications, potentially excluding uncertain predictions.
    - `--dataset FF++`: Specifies the dataset name. It uses the dataset to determine which prediction .txt files to use. It then dynamically creates and saves results and plots in appropriate directories based on this dataset name. 

   **Example Command**
   ```sh
   python results/process_prediction_values_and_compute_metrics.py --threshold 0.5 --propThreshold 0.1 --dataset FF++
   ```

3. **View Results**:
    - Evaluation metrics can be found in `results/metrics/results.txt`.
    - ROC curve plots can be found in `results/roc_plots/`.
