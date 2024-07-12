# LipForensics

This directory contains supplementary scripts and results for the evaluation of the RealForensics model. 
The results presented in my bachelor thesis can be recreated or tested on additional datasets using the provided scripts.

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

After extracting the frames in the videos and detecting the landmarks the folders have to be placed as following:


## Usage 

1. **Pretrained Model**
   - Download the [`pretrained_model`](https://drive.google.com/file/d/1wfZnxZpyNd5ouJs0LjVls7zU0N_W73L7/view?usp=sharing) on google drive
   - Place it in the models/weights folder. 

2. **Execute the evlauate.py Script to load the Pretrained Model and Compute Metrics**:
    ```sh
    python evaluate.py --dataset Celeb-DF --weights_forgery ./models/weights/lipforensics_ff.pth
    ```
   
    - the supplemented script is available at `./evaluate.py`
