import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, roc_curve, auc
import os
import argparse

# Basisverzeichnis für die Vorhersagewerte relativ zum Skriptverzeichnis
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'process_data', 'prediction_values'))

def read_predictions(file_path):
    predictions = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            try:
                score_str = line.split(':')[-1].strip().strip('[]')
                score = float(score_str)
                predictions.append(score)
            except ValueError as e:
                print(f"Skipping line due to error: {line.strip()} -> {e}")
    return predictions

def main(threshold, propThreshold, dataset):
    # Setzen der vollständigen Pfade basierend auf dem Dataset-Namen
    real_path = os.path.join(BASE_DIR, f'prediction_real_values_{dataset}.txt')
    fake_path = os.path.join(BASE_DIR, f'prediction_fake_values_{dataset}.txt')

    real_scores = read_predictions(real_path)
    fake_scores = read_predictions(fake_path)

    total_real = len(real_scores)
    total_fake = len(fake_scores)
    total = total_real + total_fake

    filtered_real_scores = [score for score in real_scores if score <= threshold - propThreshold or score >= threshold + propThreshold]
    filtered_fake_scores = [score for score in fake_scores if score <= threshold - propThreshold or score >= threshold + propThreshold]

    filtered_total = len(filtered_real_scores) + len(filtered_fake_scores)
    filtered_percentage = (filtered_total / total) * 100

    print(f"Total data points: {total}")
    print(f"Filtered data points: {filtered_total}")
    print(f"Percentage of data points categorized: {filtered_percentage:.2f}%")

    # TP, FP, TN, FN initialisieren
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # Werte in TP, FP, TN und FN einteilen
    for score in filtered_real_scores:
        if score > threshold:
            FN += 1
        else:
            TN += 1

    for score in filtered_fake_scores:
        if score > threshold:
            TP += 1
        else:
            FP += 1

    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Negatives (FN): {FN}")

    # True Labels und Predicted Scores initialisieren
    true_labels = []
    predicted_scores = []

    # True Labels und Predicted Scores füllen
    for score in filtered_real_scores:
        true_labels.append(0)  # Echte Videos sind 0
        predicted_scores.append(score)

    for score in filtered_fake_scores:
        true_labels.append(1)  # Gefälschte Videos sind 1
        predicted_scores.append(score)

    # Predicted Labels anhand des Schwellenwerts
    predicted_labels = [1 if score > threshold else 0 for score in predicted_scores]

    # Berechnung der Metriken
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    average_precision = average_precision_score(true_labels, predicted_scores)
    roc_auc = roc_auc_score(true_labels, predicted_scores)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Average Precision (AP): {average_precision:.4f}")
    print(f"Area Under the Curve (AUC): {roc_auc:.4f}")

    # Kombination der Vorhersagen und Labels
    all_predictions = np.concatenate([filtered_real_scores, filtered_fake_scores])
    all_labels = np.concatenate([[0] * len(filtered_real_scores), [1] * len(filtered_fake_scores)])

    # Berechnung der AUC-Kurve
    fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
    roc_auc = auc(fpr, tpr)

    # Verzeichnis zum Speichern der Bilder
    plot_dir = f'results/roc_plots/{dataset}'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Dateiname basierend auf Schwellenwerten
    filename = f'roc_curve_threshold_{threshold}_propThreshold_{propThreshold}.png'
    filepath = os.path.join(plot_dir, filename)

    # Zeichnen der AUC-Kurve und speichern als Bild
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic\nThreshold: {threshold}, PropThreshold: {propThreshold}')
    plt.legend(loc='lower right')
    plt.savefig(filepath)
    plt.close()

    # Speichern der Ergebnisse in eine Textdatei
    result_dir = 'results/metrics'
    os.makedirs(result_dir, exist_ok=True)
    results_file = os.path.join(result_dir, f'results_{dataset}.txt')

    with open(results_file, 'a') as f:  # 'a' für Anhängen
        f.write(f"Threshold: {threshold}, PropThreshold: {propThreshold}\n")
        f.write(f"Total data points: {total}\n")
        f.write(f"Filtered data points: {filtered_total}\n")
        f.write(f"Percentage of data points categorized: {filtered_percentage:.2f}%\n")
        f.write(f"True Positives (TP): {TP}\n")
        f.write(f"False Positives (FP): {FP}\n")
        f.write(f"True Negatives (TN): {TN}\n")
        f.write(f"False Negatives (FN): {FN}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Average Precision (AP): {average_precision:.4f}\n")
        f.write(f"Area Under the Curve (AUC): {roc_auc:.4f}\n")
        f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions with different thresholds.")
    parser.add_argument('--threshold', type=float, required=True, help='Threshold value')
    parser.add_argument('--propThreshold', type=float, required=True, help='Proportional threshold value')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset (e.g., FF++, Celeb-DF)')
    args = parser.parse_args()

    main(args.threshold, args.propThreshold, args.dataset)
