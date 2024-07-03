# BA-Deepfake-Detection-Utils

This repository contains supplementary scripts and documentation for the evaluation of three state-of-the-art deepfake detection models as part of my bachelor's thesis.

## Content of the repository

This repository provides a detailed overview of how the models were evaluated using different datasets. It includes step-by-step instructions on:

- Downloading the models and datasets
- Managing necessary dependencies
- Preparing the data
- Using the models
- Adapting the scripts
- Evaluating the performance

All necessary scripts and documentation are included to replicate the evaluation metrics and ROC curves, or to test on new datasets contained in the repository.

## Models

The respective directories and github links of the evaluated Models with supplemanty scipts and a step by step instruction 

- [`LipForensics`](./LipForensics) TODO: bsp. A CNN based approach released in the year 2020  [git](https://github.com/ahaliassos/LipForensics)
- [`icpr2020dfdc`](./icpr2020dfdc) [git](https://github.com/polimi-ispl/icpr2020dfdc)
- [`RealForensics`](./RealForensics) [git](https://github.com/ahaliassos/RealForensics)

## Datasets

- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)

Both RealForensics and LipForensics require preprocessed data. Instructions for preparing the data can be found in the [`Datasets/preprocess_datasets`](./Datasets/preprocess_datasets) directory

## Purpose

This repository is for the evaluation of three state-of-the-art deepfake detection models. It includes supplementary scripts used during the evaluation, as well as detailed instructions and documentation on the evaluation process as described in the thesis. This repository supports the reproduction of the model results of the thesis.

## Contact

If you have any questions or comments, please do not hesitate to contact me at [niklas.langes@hs-weingarten.de].
