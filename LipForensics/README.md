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

### Video Placement

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

### Facial Landmarks Placement

Place the facial landmarks in the corresponding folders as `.npy` files. Use the same directory structure as for the videos, but replace `videos` with `landmarks`. Each landmark file should have the same name as its corresponding video, except that it ends in `.npy`.

**Example** for FaceForensics++ Deepfakes landmarks:
`data/Forensics/Deepfakes/c23/landmarks`

In my case, the 68 landmarks are computed with face alignment.

## Usage 
    
    ```sh
    pip install hydra-core pytorchvideo
    ```


1. **Pretrained Model**
   - Download the [`pretrained_model`](https://drive.google.com/file/d/1nqEVlRN51WyzMWSeB7x9okcaegFgA-BQ/view) on google drive

2. **Execute the Script to load the Pretrained Model and Compute Metrics**:
    ```sh
    python stage2/eval.py model.weights_filename=realforensics_ff.pth
    ```
   
    - the supplemented script is available at `./load_pretrained_model_and_compute_metrics.py`
