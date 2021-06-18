# AnomalyDetectionCVPR2018-Pytorch
Pytorch version of - https://github.com/WaqasSultani/AnomalyDetectionCVPR2018

## Future Improvements
In this section, I list the future improvements I intend to add to this repository. Please feel free to recommend new features. I also happily accept PR's! :smirk:

* I3D feature extraction
* MFNET feature extraction

## Known Issues:

* AUC is not exactly as reported in the paper (0.70 vs 0.75) - might be affected by the weights of C3D
* Faulty video_demo

## Install Anaconda Environment
```conda env create -f environment.yml```


```conda activate adCVPR18```

## Download C3D Weights
I couldn't upload here the weights for the C3D model because the file is too big, but it can be found here:
https://github.com/DavideA/c3d-pytorch

## Precomputed Features
Can be downloaded from:
https://drive.google.com/drive/folders/1rhOuAdUqyJU4hXIhToUnh5XVvYjQiN50?usp=sharing

## Pre-Trained Anomaly Detector
Check out <a href="exps/models">exps/models</a> for for trained models on the pre-computed features

The loss graph during training is shown here:

<img src=graphs/Train_loss.png width="600"/>

## Features Extraction
```python feature_extractor.py --dataset_path "path-to-dataset" --model_type "fe-model-eg-c3d" --pretrained_3d "path-to-pretrained-fe"```

## Training
```python TrainingAnomalyDetector_public.py --features_path "path-to-dataset" --annotation_path "path-to-train-annos"```

## Generate ROC Curve
```python generate_ROC.py --features_path "path-to-dataset" --annotation_path "path-to-annos" --model_path "path-to-model"```

Using my pre-trained model after 40K iterations, I achieve this following performance on the test-set. I'm aware that the current model doesn't achieve AUC of 0.75 as reported in the original paper. This can be caused by different weights of the C3D model.

<img src=graphs/roc_auc.png width="600"/>

## Demo *
```python video_demo.py --feature_extractor "path-to-pretrained-fe" --feature_method "fe-method" --ad_model "path-to-pretrained-ad-model" --n_segments "number-of-segments"```

The GUI would let you load a video and run the Anomaly Detection code (including feature extraction) and output a video with a graph of the Anomaly Detection prediction below.

## Annotation *
```python annotation_methods.py --path_list LIST_OF_VIDEO_PATH --dir_list LIST_OF_LIST_WITH_PATH_AND_VIDEO_NAME --normal_or_not LIST_TRUE_FALUE```

This is currently just for demo but will allow training with nex videos

*Contrbuted by Peter Overbury of Sussex Universty IISP Group

## Cite
```
@misc{anomaly18cvpr-pytorch,
  author       = "Eitan Kosman",
  title        = "Pytorch implementation of Real-World Anomaly Detection in Surveillance Videos",
  howpublished = "\url{https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch}",
  note         = "Accessed: 20xx-xx-xx"
}
```

## FAQ
1. 
```
Q: video_demo doesn't show videos
A: Downlaod and install LAVFilters: http://forum.doom9.org/showthread.php?t=156191
```