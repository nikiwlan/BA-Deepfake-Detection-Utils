import argparse
from collections import defaultdict
import time
import pandas as pd
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc
import os
import matplotlib.pyplot as plt
import numpy as np
from data.transforms import NormalizeVideo, ToTensorVideo
from data.dataset_clips import ForensicsClips, CelebDFClips, DFDCClips, CustomDatasetClips
from data.samplers import ConsecutiveClipSampler
from models.spatiotemporal_net import get_model
from utils import get_files_from_split

def parse_args():
    parser = argparse.ArgumentParser(description="DeepFake detector evaluation")
    parser.add_argument(
        "--dataset",
        help="Dataset to evaluate on",
        type=str,
        choices=[
            "FaceForensics++",
            "Deepfakes",
            "FaceSwap",
            "Face2Face",
            "NeuralTextures",
            "FaceShifter",
            "DeeperForensics",
            "CelebDF",
            "CustomDataset",
            "DFDC",
        ],
        default="FaceForensics++",
    )
    parser.add_argument(
        "--compression",
        help="Video compression level for FaceForensics++",
        type=str,
        choices=["c0", "c23", "c40"],
        default="c23",
    )
    parser.add_argument("--grayscale", dest="grayscale", action="store_true")
    parser.add_argument("--rgb", dest="grayscale", action="store_false")
    parser.set_defaults(grayscale=True)
    parser.add_argument("--frames_per_clip", default=25, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--device", help="Device to put tensors on", type=str, default="cuda:0")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--weights_forgery_path",
        help="Path to pretrained weights for forgery detection",
        type=str,
        default="./models/weights/lipforensics_ff.pth"
    )
    parser.add_argument(
        "--split_path", help="Path to FF++ splits", type=str, default="./data/datasets/Forensics/splits/test.json"
    )
    parser.add_argument(
        "--dfdc_metadata_path", help="Path to DFDC metadata", type=str, default="./data/datasets/DFDC/metadata.json"
    )
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold value for adjusting positive/negative predictions.")
    parser.add_argument("--probability_threshold", type=float, default=0.0, help="Probability threshold for ignoring logits.")

    args = parser.parse_args()

    return args


def compute_with_average_of_video(video_to_logits, video_to_labels, threshold, probability_threshold):
    """ "
    Compute video-level area under ROC curve. Averages the logits across the video for non-overlapping clips.

    Parameters
    ----------
    video_to_logits : dict
        Maps video ids to list of logit values
    video_to_labels : dict
        Maps video ids to label
    """

    output_batch = torch.stack(
        [torch.mean(torch.stack(video_to_logits[video_id]), 0, keepdim=False) for video_id in video_to_logits.keys()]
    )
    output_labels = torch.stack([video_to_labels[video_id] for video_id in video_to_logits.keys()])

    fpr, tpr, _ = metrics.roc_curve(output_labels.cpu().numpy(), output_batch.cpu().numpy())

    # Berechnung der Metriken
    TP, FP, TN, FN, precision, recall, f1_score, sensitivity, ap, partital_AUC = calculate_metrics(output_labels.cpu().numpy(), output_batch.cpu().numpy(), threshold, probability_threshold)

    videos_total = TP + FP + TN + FN
    # Ausgabe der Ergebnisse
    print("-----------------------------------------------------")
    print("Durschnitt genommen mit threshold von ", threshold, " und probability threshold von ", probability_threshold, ":")
    print("-----------------------------------------------------")
    print("Video Count", videos_total)
    print("True Positives (TP): ", TP)
    print("False Positives (FP): ", FP)
    print("True Negatives (TN): ", TN)
    print("False Negatives (FN): ", FN)
    print("Percentage True Positives (TP): ", TP/videos_total*100)
    print("Percentage False Positives (FP): ", FP/videos_total*100)
    print("Percentage True Negatives (TN): ", TN/videos_total*100)
    print("Percentage False Negatives (FN): ", FN/videos_total*100)
    print("-----------------------------------------------------")
    print("Sensitivität (True Positive Rate): ", sensitivity)
    print("Präzision (Positive Predictive Value): ", precision)
    print("Recall (Sensitivity): ", recall)
    print("F1-Score: ", f1_score)
    print("AP (Average precision): ", ap)
    print("pAUC (partital AUC): ", partital_AUC)
    print("-----------------------------------------------------")
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
 
    return metrics.auc(fpr, tpr)

def calculate_metrics(y_true, y_pred, threshold, probability_threshold):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        # Ignoriere Werte innerhalb des probability_threshold-Bereichs
        if (-probability_threshold + threshold) <= y_pred[i] <= (probability_threshold + threshold) :  #             -2+2 <= 4 <= 2+2
            continue

        # Verwende threshold für die Klassifizierung
        if y_pred[i] >= threshold and y_true[i] == 1:
            TP += 1
        if y_pred[i] >= threshold and y_true[i] == 0:
            FP += 1
        if y_pred[i] < threshold and y_true[i] == 0:
            TN += 1
        if y_pred[i] < threshold and y_true[i] == 1:
            FN += 1
    
    # Sensitivity
    if TP + FN != 0:
        sensitivity = TP / (TP + FN)
    else:
        sensitivity = None

    # Precision
    if TP + FP != 0:
        precision = TP / (TP + FP)
    else:
        precision = None

    # Recall (Sensitivity)
    if TP + FN != 0:
        recall = TP / (TP + FN)
    else:
        recall = None

    # F1-Score
    if precision is not None and recall is not None and precision + recall != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = None

     # Average Precision
    ap = average_precision_score(y_true, y_pred)

    # Partial AUC (pAUC10)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    partial_auc = auc(fpr[fpr <= 0.1], tpr[fpr <= 0.1])

    
    return TP, FP, TN, FN, precision, recall, f1_score, sensitivity, ap, partial_auc


def validate_video_level(model, loader, args):
    """ "
    Evaluate model using video-level AUC score.

    Parameters
    ----------
    model : torch.nn.Module
        Model instance
    loader : torch.utils.data.DataLoader
        Loader for forgery data
    args
        Options for evaluation
    """
    model.eval()

    video_to_logits = defaultdict(list)
    video_to_labels = {}

    with torch.no_grad():
        for data in tqdm(loader):
            images, labels, video_indices = data
            images = images.to(args.device)
            labels = labels.to(args.device)       # 32 mal 0 Label

            # Forward
            logits = model(images, lengths=[args.frames_per_clip] * images.shape[0])  # Anzahl der Bilder und Batch-Größe images.shape[0]
            # args.frames_per_clip also es sind 25 frames per clip und 13 - 18 Clips pro Video je nach dem Wie lange Video ist
            # die einzelnen Clips werden auf die Wahrscheinlihkeit eingeschätzt 

            #probabilities = torch.sigmoid(logits).cpu()

            # Konvertiere den Tensor in ein Numpy-Array
            # numpy_array = probabilities.numpy()

            # Iteriere über jedes Element und multipliziere es
            # for i in range(len(numpy_array)):
                # numpy_array[i] = numpy_array[i] * 100
            # Get maps from video ids to list of logits (representing outputs for clips) as well as to label
            for i in range(len(video_indices)):
                video_id = video_indices[i].item()
                video_to_logits[video_id].append(logits[i])  
                video_to_labels[video_id] = labels[i]
 
            # ich denke er macht nur einen clip pro Video bisher !!!

    # timer
    end_time = time.time()
    threshold = args.threshold
    probability_threshold = args.probability_threshold
    auc_video = compute_with_average_of_video(video_to_logits, video_to_labels, threshold, probability_threshold)
    return auc_video, end_time


video_id = 0

percentage_fp_total =  0
percentage_tn_total =  0
percentage_tp_total =  0
percentage_fn_total =  0 

total_true = 0
total_false = 0
video_count = 0

video_fp = 0
video_tn = 0
video_tp = 0
video_fn = 0

def get_subdirectories(directory_path):
    return [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]

def main():

    # timer
    start_time = time.time()

    args = parse_args()

    model = get_model(weights_forgery_path=args.weights_forgery_path)

    # Get dataset
    transform = Compose(
        [ToTensorVideo(), CenterCrop((88, 88)), NormalizeVideo((0.421,), (0.165,))]
    )
    if args.dataset in [
        "FaceForensics++",
        "Deepfakes",
        "FaceSwap",
        "Face2Face",
        "NeuralTextures",
        "FaceShifter",
        "DeeperForensics",
    ]:
        if args.dataset == "FaceForensics++":
            fake_types = ("Deepfakes", "FaceSwap", "Face2Face", "NeuralTextures")
        else:
            fake_types = (args.dataset,)

        
        real_path = "D:\\LipForensics\\LipForensics\\data\\datasets\\Forensics\\RealFF\\c23\\images"
        fake_base_path = "D:\\LipForensics\\LipForensics\\data\\datasets\\Forensics\\"
        compression = args.compression
        test_files_real = get_subdirectories(real_path)
        test_files_fake = get_subdirectories(os.path.join(fake_base_path, fake_types[0], compression, "images"))

        dataset = ForensicsClips(
            test_files_real,
            test_files_fake,
            args.frames_per_clip,
            grayscale=args.grayscale,
            compression=args.compression,
            fakes=fake_types,
            transform=transform,
            max_frames_per_video=110,
        )
    elif args.dataset == "CelebDF":
        dataset = CelebDFClips(args.frames_per_clip, args.grayscale, transform)
    elif args.dataset == "CustomDataset":
        dataset = CustomDatasetClips(args.frames_per_clip, args.grayscale, transform)
    else:
        metadata = pd.read_json(args.dfdc_metadata_path).T
        dataset = DFDCClips(args.frames_per_clip, metadata, args.grayscale, transform)

    # Get sampler that splits video into non-overlapping clips
    sampler = ConsecutiveClipSampler(dataset.clips_per_video)

    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)

    auc, end_time = validate_video_level(model, loader, args)
    print(args.dataset, f"AUC (Mit Mitteln der Werte): {auc}")
    print("-----------------------------------------------------")
    print("time alg.: ", end_time - start_time)

if __name__ == "__main__":
    main()
