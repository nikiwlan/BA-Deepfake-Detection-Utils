RealForensics

This directory contains supplementary scripts and results for the evaluation of the icpr2020dfdc model. 
The results presented in my bachelor thesis can be recreated or tested on additional datasets using the provided scripts.

Setup

Create a virtual environment:

python3 -m venv myenv

Activate the virtual environment:

source myenv/bin/activate

Install necessary packages:

pip install torch numpy scipy argparse torchvision matplotlib

Clone the repository:

git clone https://github.com/polimi-ispl/icpr2020dfdc

Install additional dependencies:

pip install efficientnet-pytorch
pip install -U git+https://github.com/albu/albumentations > /dev/null

Navigate to the notebook directory:

cd icpr2020dfdc/notebook

Clone this repository:

git clone https://github.com/nikiwlan/BA-Deepfake-Detection-Utils.git

Navigate to the icpr2020dfdc directory:

cd BA-Deepfake-Detection-Utils/icpr2020dfdc

Directory Structure

    process_data/: Contains scripts for processing labeled .mp4 folders. There is also a sub-folder with predicted values for the FF++ and Celeb-DF datasets.
        calculate_fake_pred_values.py: Script to calculate prediction values for fake videos.
        calculate_real_pred_values.py: Script to calculate prediction values for real videos.
        prediction_values/: Contains real and fake prediction values for labeled datasets: FF++ and Celeb-DF.
            prediction_fake_values_celeb-df.txt: Prediction values for 500 fake videos from the Celeb-DF dataset.
            prediction_fake_values_ff++: Prediction values for 600 fake videos from the FaceForensics++ dataset.
            prediction_real_values_celeb-df.txt: Prediction values for 500 real videos from the Celeb-DF dataset.
            prediction_real_values_ff++: Prediction values for 600 real videos from the FaceForensics++ dataset.

    results/: Contains results and metrics from the evaluation.
        process_prediction_values_and_compute_metrics.py: Script to process prediction values and compute evaluation metrics.
        metrics/: Contains detailed metrics for the evaluated models.
            results.txt: Text file with various metrics and details on different thresholds and probability thresholds.
        roc_plots/: Contains ROC curve plots on various thresholds and probability thresholds.
            Various ROC curve plots as .png files.

Usage

Calculate Prediction Values:

    Note: Adjust the paths of the datasets in the scripts!

    For fake videos:

    python3 process_data/calculate_fake_pred_values.py --dataset FF++

    --dataset: Specifies the dataset to process. Use FF++ or Celeb-DF.

For real videos:

python3 process_data/calculate_real_pred_values.py --dataset FF++

        --dataset: Specifies the dataset to process. Use FF++ or Celeb-DF.

    Note: The newly generated prediction values are created in the icpr2020dfdc root directory!

Process Prediction Values and Compute Metrics:

python3 results/process_prediction_values_and_compute_metrics.py --threshold [value] --propThreshold [value] --real_path [path_to_real_predictions] --fake_path [path_to_fake_predictions]

    --threshold: Threshold value.
    --propThreshold: Probability threshold value.
    --real_path: Path to the .txt file with real video predictions.
    --fake_path: Path to the .txt file with fake video predictions.

Example Command

python3 results/process_prediction_values_and_compute_metrics.py --threshold 0 --propThreshold 0 --real_path prediction_real_values_ff++.txt --fake_path prediction_fake_values_ff++.txt

View Results:

    Evaluation metrics can be found in results/metrics/results.txt.
    ROC curve plots can be found in results/roc_plots/.

