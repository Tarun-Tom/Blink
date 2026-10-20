import os
import json
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

CLASS_NAMES = ["sleepy", "awake"]

def set_seeds(seed: int = 42):
    """Sets random seeds for Python, NumPy, and TensorFlow to ensure reproducible runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def save_history(history_dict: dict, file_path: str):
    """Saves training history dictionary to JSON."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean_history = {k: [float(v) for v in vals] for k, vals in history_dict.items()}
    with open(p, "w") as f:
        json.dump(clean_history, f, indent=4)

def plot_training_curves(history_dict: dict, save_path: str, model_title: str = "Model"):
    """Generates and saves accuracy and loss progression curves."""
    epochs = range(1, len(history_dict.get("loss", [])) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    ax1.plot(epochs, history_dict["loss"], "b-o", label="Training Loss")
    if "val_loss" in history_dict:
        ax1.plot(epochs, history_dict["val_loss"], "r--s", label="Validation Loss")
    ax1.set_title(f"{model_title}: Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (Binary Cross-Entropy)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy Curve
    ax2.plot(epochs, history_dict["accuracy"], "b-o", label="Training Accuracy")
    if "val_accuracy" in history_dict:
        ax2.plot(epochs, history_dict["val_accuracy"], "r--s", label="Validation Accuracy")
    ax2.set_title(f"{model_title}: Training & Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def evaluate_predictions(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5):
    """
    Computes confusion matrix, classification report, and summary statistics.
    Labels: 0 = sleepy, 1 = awake
    """
    y_pred = (y_pred_prob >= threshold).astype(int).flatten()
    y_true = y_true.astype(int).flatten()

    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, digits=4, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    
    return {
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "y_true": y_true,
        "y_pred": y_pred
    }

def plot_confusion_matrix(cm_array: list, save_path: str, model_title: str = "Model"):
    """Plots and saves confusion matrix display."""
    cm = np.array(cm_array)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title(f"{model_title} Confusion Matrix")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
