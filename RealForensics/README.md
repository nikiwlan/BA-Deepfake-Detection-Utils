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

### Dataset Preprocessing

The steps to preprocess the datasets are located in a different directory. Please follow the instructions provided in the following location:

[`Preprocess Datasets`](../Datasets/preprocess_datasets)

After extracting the frames in the videos and detecting the landmarks the folders have to be placed as following:

## Usage 

1. **Install the required package**
    ```sh
    pip install hydra-core pytorchvideo
    ```
2. **Pretrained Model**
   - Download the [`pretrained_model`](https://drive.google.com/file/d/1nqEVlRN51WyzMWSeB7x9okcaegFgA-BQ/view) on google drive
   - Place it in the stage2/weights folder. 

3. **Execute the Script to load the Pretrained Model and Compute Metrics**:
    ```sh
    python stage2/eval.py model.weights_filename=realforensics_ff.pth
    ```
   
    - the supplemented script is available at `./load_pretrained_model_and_compute_metrics.py`
