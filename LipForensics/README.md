# LipForensics

This directory contains additional scripts and step-by-step instructions for evaluating the LipForensics model. 

## Setup

1. **Clone the LipForensics git repository**:
    ```sh
    git clone https://github.com/ahaliassos/LipForensics.git
    ```

2. **Navigate to the repository**:
    ```sh
    cd LipForensics
    ```

3. **Install the packages via the requirements.txt**:
    ```sh
    pip install -r requirements.txt
    ```

### Dataset Preprocessing

The steps to preprocess the datasets are located in a different directory. Please follow the instructions provided in the following location:

[`Preprocess Datasets`](../Datasets/preprocess_datasets)


## Usage 

1. **Pretrained Model**
   - Download the [`pretrained_model`](https://drive.google.com/file/d/1wfZnxZpyNd5ouJs0LjVls7zU0N_W73L7/view?usp=sharing) on google drive
   - Place it in the models/weights folder. 

2. **Execute the evlauate.py Script to load the Pretrained Model and Compute Metrics**:
   
    - the supplemented script is available at `./evaluate.py`
      
    ```sh
    python evaluate.py --dataset FaceForensics++ --threshold 0 --probability_threshold 0 
    ```
    
    - `--dataset`: Specifies the dataset to process. Use `FaceForensics++`, `Celeb-DF` or `CustomDataset`.
    - `--threshold`: Adjusts the threshold. For example, with a value of 1, anything below 1 is considered negative and anything above is considered positive.
    - `--probability_threshold`: Ignores values within a range. For example, with a value of 1, all values between -1 and 1 are ignored (no prediction).
