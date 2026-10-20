import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import time
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_dataset, CLASS_NAMES
from src.cnn_model import build_cnn
from src.utils import (
    set_seeds,
    save_history,
    plot_training_curves,
    evaluate_predictions,
    plot_confusion_matrix
)

def train_and_evaluate_cnn(
    dataset_dir: str = "dataset",
    batch_size: int = 64,
    epochs: int = 10,
    learning_rate: float = 0.001
):
    print("=" * 70)
    print("STARTING FULL CNN BASELINE TRAINING (10 EPOCHS)")
    print("=" * 70)

    set_seeds(42)
    models_dir = Path("models")
    results_dir = Path("results")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Datasets
    print("\n[Step 1] Loading Train, Validation, and Test datasets...")
    train_ds = load_dataset(dataset_dir, split="train", batch_size=batch_size, shuffle=True, seed=42)
    val_ds = load_dataset(dataset_dir, split="val", batch_size=batch_size, shuffle=False)
    test_ds = load_dataset(dataset_dir, split="test", batch_size=batch_size, shuffle=False)

    # 2. Build CNN Model
    print("\n[Step 2] Building Compact CNN Architecture...")
    model = build_cnn(input_shape=(64, 64, 1), learning_rate=learning_rate, dropout_rate=0.3)
    model.summary()

    # Callbacks: Best validation checkpoint
    best_model_path = models_dir / "cnn_baseline_best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        )
    ]

    # 3. Train Model
    print(f"\n[Step 3] Training CNN for {epochs} epochs on native Windows CPU...")
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    training_duration = time.time() - start_time
    print(f"\nCNN Training completed in {training_duration:.2f} seconds ({training_duration/60:.2f} minutes).")

    # Save final model
    final_model_path = models_dir / "cnn_baseline_final.keras"
    model.save(final_model_path)
    print(f"Final CNN model saved to: {final_model_path}")
    print(f"Best CNN checkpoint saved to: {best_model_path}")

    # 4. Save History and Plot Curves
    history_dict = history.history
    history_file = results_dir / "cnn_training_history.json"
    save_history(history_dict, str(history_file))
    print(f"Training history saved to: {history_file}")

    curves_path = results_dir / "cnn_training_curves.png"
    plot_training_curves(history_dict, str(curves_path), model_title="CNN Baseline (64x64)")
    print(f"Training curves saved to: {curves_path}")

    # 5. Evaluate Best Checkpoint on Validation & Test Sets
    print("\n[Step 4] Evaluating Best Checkpoint on Validation & Test Splits...")
    best_model = tf.keras.models.load_model(best_model_path)

    def run_eval(dataset, split_name):
        y_true_list = []
        y_prob_list = []
        for x_batch, y_batch in dataset:
            probs = best_model(x_batch, training=False).numpy()
            y_prob_list.append(probs)
            y_true_list.append(y_batch.numpy())
            
        y_true = np.concatenate(y_true_list).flatten()
        y_prob = np.concatenate(y_prob_list).flatten()
        metrics = evaluate_predictions(y_true, y_prob, threshold=0.5)

        # Plot confusion matrix
        cm_path = results_dir / f"cnn_{split_name}_confusion_matrix.png"
        plot_confusion_matrix(metrics["confusion_matrix"], str(cm_path), model_title=f"CNN {split_name.title()}")

        summary_out = {
            "split": split_name,
            "total_samples": int(len(y_true)),
            "confusion_matrix": metrics["confusion_matrix"],
            "classification_report": metrics["classification_report"]
        }
        json_path = results_dir / f"cnn_{split_name}_metrics.json"
        with open(json_path, "w") as f:
            json.dump(summary_out, f, indent=4)

        print(f"\n--- CNN {split_name.upper()} RESULTS ---")
        print(metrics["classification_report_text"])
        print(f"Saved: {cm_path} and {json_path}")
        return summary_out

    val_results = run_eval(val_ds, "val")
    test_results = run_eval(test_ds, "test")

    # 6. Generate FNN vs CNN Comparative Report
    print("\n[Step 5] Compiling FNN vs CNN Comparative Performance Report...")
    fnn_test_path = results_dir / "fnn_test_metrics.json"
    fnn_history_path = results_dir / "fnn_training_history.json"
    fnn_model_path = models_dir / "fnn_baseline_best.keras"
    cnn_model_path = best_model_path

    if fnn_test_path.exists():
        with open(fnn_test_path) as f:
            fnn_test_data = json.load(f)
        fnn_rep = fnn_test_data["classification_report"]
        cnn_rep = test_results["classification_report"]

        fnn_size_mb = fnn_model_path.stat().st_size / (1024 * 1024)
        cnn_size_mb = cnn_model_path.stat().st_size / (1024 * 1024)

        comparison_data = {
            "fnn": {
                "parameters": 2262017,
                "model_size_mb": round(fnn_size_mb, 2),
                "test_accuracy": round(fnn_rep["accuracy"] * 100, 2),
                "macro_f1": round(fnn_rep["macro avg"]["f1-score"], 4),
                "sleepy_recall": round(fnn_rep["sleepy"]["recall"] * 100, 2),
                "awake_recall": round(fnn_rep["awake"]["recall"] * 100, 2),
                "training_time_sec": 171.03
            },
            "cnn": {
                "parameters": model.count_params(),
                "model_size_mb": round(cnn_size_mb, 2),
                "test_accuracy": round(cnn_rep["accuracy"] * 100, 2),
                "macro_f1": round(cnn_rep["macro avg"]["f1-score"], 4),
                "sleepy_recall": round(cnn_rep["sleepy"]["recall"] * 100, 2),
                "awake_recall": round(cnn_rep["awake"]["recall"] * 100, 2),
                "training_time_sec": round(training_duration, 2)
            },
            "dataset_split_limitation_disclosure": (
                "Both models were trained and evaluated on the same 60/20/20 random image-level split. "
                "All 37 subjects are present across train, validation, and test splits. "
                "Reported metrics reflect recognition on familiar subjects rather than unseen zero-shot subjects."
            )
        }

        with open(results_dir / "model_comparison.json", "w") as f:
            json.dump(comparison_data, f, indent=4)
        print("Comparison summary written to results/model_comparison.json")

    print("\n" + "=" * 70)
    print("CNN TRAINING, EVALUATION & COMPARISON COMPLETED")
    print("=" * 70)
    return {
        "training_duration": training_duration,
        "val_results": val_results,
        "test_results": test_results
    }

if __name__ == "__main__":
    train_and_evaluate_cnn()
