# AnomalyDetectionCVPR2018-Pytorch
Pytorch version of - https://github.com/WaqasSultani/AnomalyDetectionCVPR2018

## Install anaconda env
```conda env create -f environment.yml```


```conda activate torch```

## Download C3D weights
I couldn't upload here the weights for the C3D model because the file is too big, but it can be found here:
https://github.com/DavideA/c3d-pytorch

## Precomputed features
Can be downloaded from:
https://drive.google.com/drive/folders/1rZn-UHM_EcIXauJ0wRysQbh0mHQoNrfY?usp=sharing

## Features extraction
```python feature_extractor.py --dataset_path "path-to-dataset" --annotation_path "path-to-train-annos" --annotation_path_test "path-to-test-annos" --pretrained_3d "path-to-pretrained-c3d"```

## Training
```python TrainingAnomalyDetector_public.py --features_path "path-to-dataset" --annotation_path "path-to-train-annos" --annotation_path_test "path-to-test-annos"```

## Generate ROC curve
```python generate_ROC.py --features_path "path-to-dataset" --annotation_path "path-to-annos"```

![ROC](graphs/roc_auc.png)

## Demo*
"video_demo.py --video_parth_list LIST_OF_VIDEO_PATHS --model_dir PATH_TO_MODLE "
This should take any video and run the Anomaly Detection code (including CD3 feature extraction) and output a video with a graph of the Anomaly Detection prediction on the right-hand side (like in the demo code for the paper). It is all still a bit rough but it works and I do plan to add to it so I can always update later.

## Annotation*
"annotation_methods.py --path_list LIST_OF_VIDEO_PATH --dir_list LIST_OF_LIST_WITH_PATH_AND_VIDEO_NAME --normal_or_not LIST_TRUE_FALUE"
This is currently just for demo but will allow training with nex videos

*Contrbuted by Peter Overbury of Sussex Universty IISP Group
