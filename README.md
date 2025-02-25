![GitHub forks](https://img.shields.io/github/forks/ekosman/AnomalyDetectionCVPR2018-Pytorch?style=social)
![GitHub Repo stars](https://img.shields.io/github/stars/ekosman/AnomalyDetectionCVPR2018-Pytorch?style=social)

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ekosman/AnomalyDetectionCVPR2018-Pytorch/main.svg)](https://results.pre-commit.ci/latest/github/ekosman/AnomalyDetectionCVPR2018-Pytorch/main)
[![Lint Status](https://github.com/pycqa/isort/workflows/Lint/badge.svg?branch=develop)](https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch/actions?query=workflow%3A)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

# AnomalyDetectionCVPR2018-Pytorch  <!-- omit in toc -->
Pytorch version of - https://github.com/WaqasSultani/AnomalyDetectionCVPR2018

## Table of Contents  <!-- omit in toc -->
- [Future Improvements](#future-improvements)
- [Known Issues](#known-issues)
- [Install Anaconda Environment](#install-anaconda-environment)
- [Feature Extractor Weights](#feature-extractor-weights)
  - [C3D](#c3d)
  - [R3D-101](#r3d-101)
  - [R3D-152](#r3d-152)
- [Precomputed Features](#precomputed-features)
  - [C3D features](#c3d-features)
  - [ResNet-101 features (by @Daniele Mascali)](#resnet-101-features-by-daniele-mascali)
  - [ResNet-152 features (by @Henryy-rs)](#resnet-152-features-by-henryy-rs)
- [Pre-Trained Anomaly Detector](#pre-trained-anomaly-detector)
- [Features Extraction](#features-extraction)
- [Training](#training)
- [Generate ROC Curve](#generate-roc-curve)
- [Demo](#demo)
  - [Off-line (with video loader)](#off-line-with-video-loader)
  - [On-line (via webcam)](#on-line-via-webcam)
- [Cite](#cite)
- [FAQ](#faq)

## Future Improvements
In this section, I list the future improvements I intend to add to this repository. Please feel free to recommend new features. I also happily accept PR's! :smirk:

* I3D feature extraction
* MFNET feature extraction

## Known Issues

* AUC is not exactly as reported in the paper (0.70 vs 0.75) - might be affected by the weights of C3D

## Install Anaconda Environment

```
conda create --name adCVPR18 python=3.9 -y
conda activate adCVPR18
pip install -r requirements.txt
```

## Feature Extractor Weights

### C3D
I couldn't upload here the weights for the C3D model because the file is too big, but it can be found here:
https://github.com/DavideA/c3d-pytorch

```
# OR download the weights directly

cd AnomalyDetectionCVPR2018-Pytorch/pretrained
wget http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle
cd ..
```

### R3D-101
https://drive.google.com/file/d/1p80RJsghFIKBSLKgtRG94LE38OGY5h4y/view?usp=share_link

### R3D-152
https://drive.google.com/file/d/1irIdC_v7wa-sBpTiBlsMlS7BYNdj4Gr7/view?usp=share_link

## Precomputed Features
Can be downloaded from:

### C3D features
https://drive.google.com/drive/folders/1rhOuAdUqyJU4hXIhToUnh5XVvYjQiN50?usp=sharing

### ResNet-101 features (by @Daniele Mascali)
https://drive.google.com/file/d/1kQAvOhtL-sGadblfd3NmDirXq8vYQPvf/view?usp=sharing

### ResNet-152 features (by @Henryy-rs)
https://drive.google.com/file/d/17wdy_DS9UY37J9XTV5XCLqxOFgXiv3ZK/view

## Pre-Trained Anomaly Detector
Check out <a href="exps/">exps/</a> for for trained models on the pre-computed features

## Features Extraction
Download the dataset from: https://github.com/WaqasSultani/AnomalyDetectionCVPR2018
Arguments:
* dataset_path - path to the directory containing videos to extract features for (the dataset is available for download above)
* model_type - which type of model to use for feature extraction (necessary in order to choose the correct pre-processing)
* pretrained_3d - path to the 3D model to use for feature extraction

```python feature_extractor.py --dataset_path "path-to-dataset" --model_type "fe-model-eg-c3d" --pretrained_3d "path-to-pretrained-fe"```

## Training
Arguments:
* features_path - path to the directory containing the extracted features (pre-computed features are available for download above, or supply your own features extracted from the previous stage)
* annotation_path - path to the annotations file (Available in this repository as `Train_annotations.txt`)

```python TrainingAnomalyDetector_public.py --features_path "path-to-dataset" --annotation_path "path-to-train-annos"```

## Generate ROC Curve
Arguments:
* features_path - path to the directory containing the extracted features (pre-computed features are available for download above, or supply your own features extracted from the previous stage)
* annotation_path - path to the annotations file (Available in this repository as `Test_annotations.txt`)
* model_path - path to the trained anomaly detection model

```python generate_ROC.py --features_path "path-to-dataset" --annotation_path "path-to-annos" --model_path "path-to-model"```

I achieve this following performance on the test-set. I'm aware that the current C3D model achieves AUC of 0.69 which is worse than the original paper. This can be caused by different weights of the C3D model or usage of a different feature extractor.

| C3D (<a href="exps\c3d\models\epoch_80000.pt">Link</a>) | R3D101 (<a href="exps\resnet_r3d101_KM_200ep\models\epoch_10.pt">Link</a>) | R3D152 (<a href="exps\resnet_r3d152_KM_200ep\models\epoch_10.pt">Link</a>) |
| :-----------------------------------------------------: | :------------------------------------------------------------------------: | :------------------------------------------------------------------------: |
|      <img src=graphs/roc_auc_c3d.png width="300"/>      |               <img src=graphs/roc_auc_r101.png width="300"/>               |               <img src=graphs/roc_auc_r152.png width="300"/>               |


## Demo

### Off-line (with video loader)
Arguments:
* feature_extractor - path to the 3D model to use for feature extraction
* feature_method - which type of model to use for feature extraction (necessary in order to choose the correct pre-processing)
* ad_model - path to the trained anomaly detection model
* n_segments - the number of segments to chunk the video to (the original paper uses 32 segments)

```python video_demo.py --feature_extractor "path-to-pretrained-fe" --feature_method "fe-method" --ad_model "path-to-pretrained-ad-model" --n_segments "number-of-segments"```

The GUI lets you load a video and run the Anomaly Detection code (including feature extraction) and output a video with a graph of the Anomaly Detection prediction below.

**Note**: The feature extractor and the anomaly detection model must match. Make sure you are using the anomaly detector that was training with the corresponding features.

### On-line (via webcam)
Arguments:
* feature_extractor - path to the 3D model to use for feature extraction
* feature_method - which type of model to use for feature extraction (necessary in order to choose the correct pre-processing)
* ad_model - path to the trained anomaly detection model
* clip_length - the length of each video clip (in frames)

```python AD_live_prediction.py --feature_extractor "path-to-pretrained-fe" --feature_method "fe-method" --ad_model "path-to-pretrained-ad-model" --clip_length "number-of-frames"```

The GUI lets you load a video and run the Anomaly Detection code (including feature extraction) and output a video with a graph of the Anomaly Detection prediction below.

**Note**: The feature extractor and the anomaly detection model must match. Make sure you are using the anomaly detector that was training with the corresponding features.

*Contrbuted by Peter Overbury of Sussex Universty IISP Group

## Cite

Go to the main page of the repository and click on the citation button, where you can find the proper citation format, or:

BibTeX
```
@software{Kosman_Pytorch_implementation_of_2022,
author = {Kosman, Eitan},
month = jan,
title = {{Pytorch implementation of Real-World Anomaly Detection in Surveillance Videos}},
url = {https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch},
version = {1.0.0},
year = {2022}
}
```

APA
```
Kosman, E. (2022). Pytorch implementation of Real-World Anomaly Detection in Surveillance Videos (Version 1.0.0) [Computer software]. https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch
```



## FAQ
1.
```
Q: video_demo doesn't show videos
A: Downlaod and install LAVFilters: http://forum.doom9.org/showthread.php?t=156191
```

2.
```
Q: What is the meaning of the second column of Train_Annotations.txt?
A: Length of the video in frames. Note that it has not effect on training. It exists because these are the original annotations supplied by the authors.
```
