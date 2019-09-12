# AnomalyDetectionCVPR2018-Pytorch
Pytorch version of - https://github.com/WaqasSultani/AnomalyDetectionCVPR2018


## Features extraction
python feature_extracor.py --dataset_path "path-to-dataset" --annotation_path "path-to-train-annos" --annotation_path_test "path-to-test-annos" --pretrained_3d "path-to-pretrained-c3d"

## Training
python TrainingAnomalyDetector_public.py --features_path "path-to-dataset" --annotation_path "path-to-train-annos" --annotation_path_test "path-to-test-annos"

## Generate ROC curve
python generate_ROC.py --features_path "path-to-dataset" --annotation_path "path-to-annos"

![ROC](graphs/roc_auc.png)
