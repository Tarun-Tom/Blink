import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def evaluate_webcam_samples(
    model_path: str = "models/cnn_baseline_best.keras",
    samples_dir: str = "results/diagnostics/manual_samples",
    output_report: str = "results/diagnostics/webcam_evaluation_report.json"
):
    print("=" * 70)
    print("WEBCAM EYE-STATE SAMPLE EVALUATION & DOMAIN-GAP BENCHMARK")
    print("=" * 70)

    model_file = Path(model_path).resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_file}")

    print(f"Loading trained CNN model: {model_file.name}...")
    model = tf.keras.models.load_model(str(model_file))

    samples_path = Path(samples_dir)
    open_dir = samples_path / "open"
    closed_dir = samples_path / "closed"
    metadata_log = samples_path / "sample_metadata.jsonl"

    open_files = sorted([f for f in open_dir.glob("*.png") if not f.name.endswith("_frame.png")])
    closed_files = sorted([f for f in closed_dir.glob("*.png") if not f.name.endswith("_frame.png")])

    print(f"\nDiscovered Labeled Crops:")
    print(f"  OPEN (Awake, Class 1):   {len(open_files)} crops")
    print(f"  CLOSED (Sleepy, Class 0): {len(closed_files)} crops")

    total_samples = len(open_files) + len(closed_files)
    if total_samples == 0:
        print("\n[NOTE] No labeled crops found in results/diagnostics/manual_samples/.")
        print("To collect real-time webcam samples:")
        print("  1. Run: .venv\\Scripts\\python.exe src/webcam_detector.py")
        print("  2. Press [O] while looking at camera with eyes OPEN.")
        print("  3. Press [C] with eyes CLOSED.")
        print("  4. Press [Q] to quit and re-run this evaluation script.")
        return None

    # Load metadata entries if available
    metadata_by_ts = {}
    if metadata_log.exists():
        with open(metadata_log, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        metadata_by_ts[entry.get("timestamp")] = entry
                    except Exception:
                        pass

    # Evaluate OPEN samples (Ground Truth = 1)
    open_results = []
    for f in open_files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        tensor = np.expand_dims(resized, axis=(0, -1)).astype(np.float32)
        p = float(model(tensor, training=False).numpy()[0][0])
        pred_label = 1 if p >= 0.5 else 0
        open_results.append({
            "file": f.name,
            "p_awake": p,
            "pred": pred_label,
            "correct": pred_label == 1
        })

    # Evaluate CLOSED samples (Ground Truth = 0)
    closed_results = []
    for f in closed_files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        tensor = np.expand_dims(resized, axis=(0, -1)).astype(np.float32)
        p = float(model(tensor, training=False).numpy()[0][0])
        pred_label = 1 if p >= 0.5 else 0
        closed_results.append({
            "file": f.name,
            "p_awake": p,
            "pred": pred_label,
            "correct": pred_label == 0
        })

    # Calculate Metrics
    tp = sum(1 for r in open_results if r["pred"] == 1)      # Ground truth OPEN, Pred OPEN
    fn = sum(1 for r in open_results if r["pred"] == 0)      # Ground truth OPEN, Pred CLOSED
    tn = sum(1 for r in closed_results if r["pred"] == 0)    # Ground truth CLOSED, Pred CLOSED
    fp = sum(1 for r in closed_results if r["pred"] == 1)    # Ground truth CLOSED, Pred OPEN (False Open)

    total_eval = tp + fn + tn + fp
    accuracy = ((tp + tn) / total_eval) * 100 if total_eval > 0 else 0.0
    open_recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    closed_recall = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0.0

    mean_p_open = float(np.mean([r["p_awake"] for r in open_results])) if open_results else 0.0
    mean_p_closed = float(np.mean([r["p_awake"] for r in closed_results])) if closed_results else 0.0

    print("\n" + "-" * 70)
    print("CLASSIFICATION PERFORMANCE SUMMARY ON WEBCAM SAMPLES:")
    print("-" * 70)
    print(f"Total Evaluated Crops: {total_eval}")
    print(f"Overall Accuracy:      {accuracy:.2f}%")
    print(f"OPEN (Awake) Recall:   {open_recall:.2f}% ({tp}/{tp+fn} correct | Mean P(awake): {mean_p_open:.4f})")
    print(f"CLOSED (Sleepy) Recall:{closed_recall:.2f}% ({tn}/{tn+fp} correct | Mean P(awake): {mean_p_closed:.4f})")
    print(f"\nConfusion Matrix (Rows: Ground Truth, Cols: Predicted):")
    print(f"                  Pred CLOSED (0)   Pred OPEN (1)")
    print(f"  Actual CLOSED:  {tn:15d}   {fp:13d}")
    print(f"  Actual OPEN:    {fn:15d}   {tp:13d}")

    # Inspect domain gap
    domain_gap_identified = False
    findings = []
    if closed_recall < 85.0 and len(closed_results) > 0:
        domain_gap_identified = True
        findings.append(
            f"Domain Shift Detected: Closed eyes scored mean P(awake)={mean_p_closed:.4f}. "
            f"{fp}/{tn+fp} closed crops were misclassified as OPEN (P >= 0.5)."
        )
    else:
        findings.append("No critical domain shift detected on evaluated samples.")

    report = {
        "model_checkpoint": str(model_file),
        "total_evaluated_crops": total_eval,
        "sample_counts": {
            "open": len(open_results),
            "closed": len(closed_results)
        },
        "accuracy_percent": round(accuracy, 2),
        "open_recall_percent": round(open_recall, 2),
        "closed_recall_percent": round(closed_recall, 2),
        "mean_p_awake_open": round(mean_p_open, 4),
        "mean_p_awake_closed": round(mean_p_closed, 4),
        "confusion_matrix": {
            "true_positive_open": tp,
            "false_negative_open": fn,
            "true_negative_closed": tn,
            "false_positive_closed_as_open": fp
        },
        "domain_gap_identified": domain_gap_identified,
        "findings": findings
    }

    out_path = Path(output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\nSaved evaluation report to: {out_path}")
    print("=" * 70)
    return report

if __name__ == "__main__":
    evaluate_webcam_samples()
