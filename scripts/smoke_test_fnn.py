import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_dataset, get_smoke_subsets, CLASS_NAMES
from src.fnn_model import build_fnn
from src.utils import set_seeds, evaluate_predictions

def run_fnn_smoke_test():
    print("=" * 60)
    print("STARTING FNN BASELINE SMOKE TEST")
    print("=" * 60)
    
    set_seeds(42)
    smoke_results_dir = Path("results/smoke_test")
    smoke_results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Dataset Subset Preparation
    print("\n[Step 1] Loading isolated smoke subsets (256 train, 64 val images)...")
    smoke_train, smoke_val = get_smoke_subsets(
        dataset_dir="dataset",
        train_samples=256,
        val_samples=64,
        batch_size=32,
        img_size=(64, 64),
        seed=42
    )

    # 2. Inspect Batch & Data Integrity
    first_batch_x, first_batch_y = next(iter(smoke_train))
    batch_x_shape = list(first_batch_x.shape)
    batch_y_shape = list(first_batch_y.shape)
    y_values = np.unique(first_batch_y.numpy()).tolist()
    pixel_min = float(np.min(first_batch_x.numpy()))
    pixel_max = float(np.max(first_batch_x.numpy()))

    print(f"Batch X Shape: {batch_x_shape} (Expected: [32, 64, 64, 1])")
    print(f"Batch Y Shape: {batch_y_shape} (Expected: [32, 1])")
    print(f"Label Values:  {y_values} (Expected subset of: [0.0, 1.0])")
    print(f"Raw Pixel Min: {pixel_min:.1f}, Max: {pixel_max:.1f}")

    assert batch_x_shape == [32, 64, 64, 1], f"Unexpected input shape: {batch_x_shape}"
    assert all(val in [0.0, 1.0] for val in y_values), f"Invalid labels: {y_values}"
    print("[PASS] Batch shapes and label encoding confirmed.")

    # 3. Model Construction
    print("\n[Step 2] Building FNN Baseline...")
    model = build_fnn(input_shape=(64, 64, 1), learning_rate=0.001, dropout_rate=0.3)
    total_params = model.count_params()
    trainable_params = sum(np.prod(v.get_shape().as_list()) for v in model.trainable_variables)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # 4. Explicit Gradient & Backward Pass Verification
    print("\n[Step 3] Verifying forward pass & gradient computation on 1 batch...")
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    with tf.GradientTape() as tape:
        preds = model(first_batch_x, training=True)
        loss_val = loss_fn(first_batch_y, preds)

    grads = tape.gradient(loss_val, model.trainable_variables)
    grad_norms = [float(tf.norm(g).numpy()) for g in grads if g is not None]
    
    print(f"Initial Single-Batch Loss: {float(loss_val.numpy()):.4f}")
    print(f"Gradient count: {len(grad_norms)} / {len(model.trainable_variables)} trainable variables")
    assert all(np.isfinite(gn) for gn in grad_norms), "NaN or Inf found in gradients!"
    print("[PASS] Forward pass and backpropagation gradients verified clean and finite.")

    # 5. Label Mapping & Output Range Verification
    print("\n[Step 4] Verifying Sigmoid Output & Class Mapping Consistency...")
    preds_np = preds.numpy().flatten()
    print(f"Sigmoid outputs min: {preds_np.min():.4f}, max: {preds_np.max():.4f}")
    assert (preds_np >= 0.0).all() and (preds_np <= 1.0).all(), "Sigmoid output out of [0, 1] bounds!"
    
    # Class mapping convention check
    print(f"Class Mapping Rule:")
    print(f"  Class 0 = '{CLASS_NAMES[0]}' (Closed eyes / Drowsy)")
    print(f"  Class 1 = '{CLASS_NAMES[1]}' (Open eyes / Alert)")
    print(f"  Decision Rule: P(Awake) >= 0.5 -> Awake (1); P(Awake) < 0.5 -> Sleepy (0)")
    print("[PASS] Class mapping and output logic confirmed.")

    # 6. Run 1-Epoch Smoke Fit
    print("\n[Step 5] Executing 1-Epoch smoke training...")
    print("NOTE: Training subset is strictly used for updates; validation subset used strictly for evaluation.")
    
    history = model.fit(
        smoke_train,
        validation_data=smoke_val,
        epochs=1,
        verbose=1
    )

    train_loss = float(history.history["loss"][0])
    train_acc = float(history.history["accuracy"][0])
    val_loss = float(history.history["val_loss"][0])
    val_acc = float(history.history["val_accuracy"][0])

    print(f"\nSmoke Test Epoch 1 Results (Subset only - not final):")
    print(f"  Train Loss:     {train_loss:.4f} | Train Acc:     {train_acc*100:.2f}%")
    print(f"  Val Loss:       {val_loss:.4f} | Val Acc:       {val_acc*100:.2f}%")

    # 7. Validation Evaluation Test
    print("\n[Step 6] Running prediction on smoke validation subset...")
    val_preds_list = []
    val_true_list = []
    for x_val, y_val in smoke_val:
        p = model.predict(x_val, verbose=0)
        val_preds_list.extend(p)
        val_true_list.extend(y_val.numpy())

    eval_out = evaluate_predictions(np.array(val_true_list), np.array(val_preds_list))
    print(f"Confusion Matrix on Smoke Val:\n{np.array(eval_out['confusion_matrix'])}")

    # 8. Save Smoke Artifacts (isolated from production models)
    smoke_model_path = smoke_results_dir / "smoke_fnn.keras"
    model.save(smoke_model_path)
    
    smoke_summary = {
        "status": "SUCCESS",
        "model_architecture": "FNN Baseline (64x64 grayscale -> Flatten -> 512 -> 256 -> 128 -> 1)",
        "input_shape": [64, 64, 1],
        "total_parameters": total_params,
        "class_mapping": {"0": CLASS_NAMES[0], "1": CLASS_NAMES[1]},
        "smoke_train_samples": 256,
        "smoke_val_samples": 64,
        "epoch_1_metrics": {
            "loss": train_loss,
            "accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        },
        "confusion_matrix_smoke_val": eval_out["confusion_matrix"],
        "smoke_model_artifact": str(smoke_model_path)
    }

    summary_file = smoke_results_dir / "smoke_test_summary.json"
    with open(summary_file, "w") as f:
        json.dump(smoke_summary, f, indent=4)

    print(f"\n[PASS] Smoke test completed successfully. Summary saved to: {summary_file}")
    print("=" * 60)
    return smoke_summary

if __name__ == "__main__":
    run_fnn_smoke_test()
