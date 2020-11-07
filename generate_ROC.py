import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from network.anomaly_detector_model import AnomalyDetector
from features_loader import FeaturesLoaderVal
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from os import path
import numpy as np
import pytorch_wrapper as pw


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")
    parser.add_argument('--features_path', default='../anomaly_features',
                        help="path to features")
    parser.add_argument('--annotation_path', default="Test_Annotation.txt",
                        help="path to annotations")
    parser.add_argument('--random-seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--model-path', type=str, default="./exps/model.weights",
                        help="set logging file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    data_loader = FeaturesLoaderVal(features_path=args.features_path,
                                    annotation_path=args.annotation_path)

    data_iter = torch.utils.data.DataLoader(data_loader,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0,  # 4, # change this part accordingly
                                            pin_memory=True)

    network = AnomalyDetector()
    system = pw.System(model=network, device=device)
    system.load_model_state(args.model_path)
    model = system.model.eval()

    # enable cudnn tune
    cudnn.benchmark = True

    y_trues = torch.tensor([])
    y_preds = torch.tensor([])

    with torch.no_grad():
        for features, start_end_couples, lengths in tqdm(data_iter):
            # features is a batch where each item is a tensor of 32 4096D features
            features = features.to(device)
            outputs = model(features).squeeze(-1)  # (batch_size, 32)
            for vid_len, couples, output in zip(lengths, start_end_couples, outputs.cpu().numpy()):
                y_true = np.zeros(vid_len)
                y_pred = np.zeros(vid_len)

                segments_len = vid_len // 32
                for couple in couples:
                    if couple[0] != -1:
                        y_true[couple[0]: couple[1]] = 1

                for i in range(32):
                    segment_start_frame = i * segments_len
                    segment_end_frame = (i + 1) * segments_len
                    y_pred[segment_start_frame: segment_end_frame] = output[i]

                if y_trues is None:
                    y_trues = y_true
                    y_preds = y_pred
                else:
                    y_trues = np.concatenate([y_trues, y_true])
                    y_preds = np.concatenate([y_preds, y_pred])

    fpr, tpr, thresholds = roc_curve(y_true=y_trues, y_score=y_preds, pos_label=1)

    plt.figure()
    lw = 2
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    os.makedirs('graphs', exist_ok=True)
    plt.savefig(path.join('graphs', 'roc_auc.png'))
