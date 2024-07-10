import logging
import os
import sys
import hydra
from pytorch_lightning import Trainer, seed_everything
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory of stage2 to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2.combined_learner import CombinedLearner
from stage2.data.combined_dm import DataModule

# static vars
logging.getLogger("lightning").propagate = False
__spec__ = None

def compute_metrics(y_true, y_pred, y_prob):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    ap = average_precision_score(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AP: {ap:.4f}")
    print(f"AUC: {auc:.4f}")

    return precision, recall, f1, ap, auc

def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('/mnt/data/roc_curve.png')
    plt.show()

@hydra.main(config_path="conf", config_name="config_combined")
def main(cfg):
    cfg.gpus = torch.cuda.device_count()
    if cfg.gpus < 2:
        cfg.trainer.accelerator = None

    learner = CombinedLearner(cfg)
    data_module = DataModule(cfg, root=os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    if cfg.model.weights_filename:
        weights_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "stage2", "weights", "realforensics_ff")
        df_weights_path = os.path.join(weights_dir, cfg.model.weights_filename)

        os.makedirs(weights_dir, exist_ok=True)

        print("Trying to load weights from: ", df_weights_path)
        if not os.path.exists(df_weights_path):
            raise FileNotFoundError(f"Weights file not found: {df_weights_path}")

        state_dict = torch.load(df_weights_path)
        weights_backbone = {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("backbone")}
        learner.model.backbone.load_state_dict(weights_backbone)
        weights_df_head = {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("df_head")}
        learner.model.df_head.load_state_dict(weights_df_head)
        print("Weights loaded.")

    trainer = Trainer(**cfg.trainer)
    results = trainer.test(learner, datamodule=data_module)
    
    # Assuming the test dataloader returns both labels and predictions
    y_true = []
    y_pred = []
    y_prob = []

    for batch in data_module.test_dataloader():
        inputs, labels = batch
        outputs = learner(inputs)
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds)
        y_prob.extend(probs)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Compute and print metrics
    precision, recall, f1, ap, auc = compute_metrics(y_true, y_pred, y_prob)
    
    # Plot ROC curve
    plot_roc_curve(y_true, y_prob)

if __name__ == "__main__":
    seed_everything(42)
    main()
