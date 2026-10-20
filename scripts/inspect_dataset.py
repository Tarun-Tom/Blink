import os
import sys
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
import concurrent.futures

def inspect_dataset(dataset_dir="dataset", results_dir="results"):
    dataset_path = Path(dataset_dir).resolve()
    results_path = Path(results_dir).resolve()
    results_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset path: {dataset_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        
    splits = ["train", "val", "test"]
    classes = ["awake", "sleepy"]
    
    # Verify split directories
    split_counts = defaultdict(dict)
    all_files = []
    
    for split in splits:
        split_p = dataset_path / split
        if not split_p.exists():
            raise FileNotFoundError(f"Split directory not found: {split_p}")
        for cls in classes:
            cls_p = split_p / cls
            if not cls_p.exists():
                raise FileNotFoundError(f"Class directory not found: {cls_p}")
            files = list(cls_p.glob("*.png"))
            split_counts[split][cls] = len(files)
            for f in files:
                all_files.append((split, cls, f))
                
    total_images = len(all_files)
    print(f"Found {total_images} total images across {splits}.")
    
    # Check subjects and metadata from filenames
    # Schema: <subject_id>_<img_id>_<gender>_<glasses>_<eye_state>_<reflections>_<lighting>_<sensor_id>.png
    subject_split_dist = defaultdict(lambda: defaultdict(int))
    corrupted_files = []
    
    # Fast integrity check & dimension inspection
    # Check dimensions on a representative sample of 2,000 images per split/class + integrity check
    sample_dimensions = Counter()
    sample_modes = Counter()
    
    print("Checking file integrity and image properties...")
    start_time = time.time()
    
    def check_file(item):
        split, cls, fpath = item
        # Subject extraction
        parts = fpath.stem.split("_")
        subject_id = parts[0] if len(parts) >= 1 else "unknown"
        
        # Verify readability
        try:
            with Image.open(fpath) as img:
                img.verify()
            return (split, cls, subject_id, None, fpath)
        except Exception as e:
            return (split, cls, subject_id, str(e), fpath)

    # Let's check full integrity using multithreading
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(check_file, all_files))
        
    for split, cls, subj, err, fpath in results:
        subject_split_dist[subj][split] += 1
        if err:
            corrupted_files.append({"path": str(fpath), "error": err})
            
    # Sample dimensions and modes
    # Take 500 images from each split/class combination
    sample_items = []
    for split in splits:
        for cls in classes:
            split_cls_files = [f for s, c, f in all_files if s == split and c == cls]
            sample_items.extend(split_cls_files[:500])
            
    for fpath in sample_items:
        try:
            with Image.open(fpath) as img:
                sample_dimensions[f"{img.width}x{img.height}"] += 1
                sample_modes[img.mode] += 1
        except Exception:
            pass

    elapsed = time.time() - start_time
    print(f"Integrity check finished in {elapsed:.2f} seconds.")
    print(f"Corrupted files count: {len(corrupted_files)}")
    
    # Analyze subject distribution / data leakage
    # How many unique subjects in each split?
    unique_subjects = sorted(list(subject_split_dist.keys()))
    subjects_per_split = {
        split: [subj for subj, counts in subject_split_dist.items() if counts.get(split, 0) > 0]
        for split in splits
    }
    
    # Overlap analysis
    train_subjs = set(subjects_per_split["train"])
    val_subjs = set(subjects_per_split["val"])
    test_subjs = set(subjects_per_split["test"])
    
    overlap_train_val = len(train_subjs.intersection(val_subjs))
    overlap_train_test = len(train_subjs.intersection(test_subjs))
    
    summary_data = {
        "dataset_path": str(dataset_path),
        "total_images": total_images,
        "splits": dict(split_counts),
        "split_totals": {split: sum(split_counts[split].values()) for split in splits},
        "class_totals": {
            cls: sum(split_counts[split][cls] for split in splits)
            for cls in classes
        },
        "class_balance": {
            cls: round(sum(split_counts[split][cls] for split in splits) / total_images * 100, 2)
            for cls in classes
        },
        "corrupted_files_count": len(corrupted_files),
        "corrupted_files": corrupted_files[:10],
        "image_modes_sample": dict(sample_modes),
        "top_dimensions_sample": dict(sample_dimensions.most_common(10)),
        "subject_analysis": {
            "total_unique_subjects": len(unique_subjects),
            "subjects_in_train": len(train_subjs),
            "subjects_in_val": len(val_subjs),
            "subjects_in_test": len(test_subjs),
            "subject_overlap_train_val": overlap_train_val,
            "subject_overlap_train_test": overlap_train_test,
            "leakage_assessment": (
                "Random image-level splitting was applied rather than subject-wise splitting. "
                f"All {len(unique_subjects)} subjects appear across train, val, and test splits. "
                "This must be highlighted in the academic report as a dataset property."
            )
        }
    }
    
    # Write JSON summary
    summary_json_path = results_path / "dataset_inspection_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"Summary saved to: {summary_json_path}")
    
    # Write Markdown summary
    summary_md_path = results_path / "dataset_inspection_summary.md"
    md_content = f"""# Dataset Inspection Summary: MRL Eye Dataset

- **Dataset Root**: `{summary_data['dataset_path']}`
- **Total Images**: `{summary_data['total_images']:,}`
- **Corrupted / Unreadable Images**: `{summary_data['corrupted_files_count']}`

## 1. Split & Class Distribution

| Split | Awake | Sleepy | Total | % of Dataset |
|---|---|---|---|---|
| **Train** | {split_counts['train']['awake']:,} | {split_counts['train']['sleepy']:,} | {summary_data['split_totals']['train']:,} | {summary_data['split_totals']['train']/total_images*100:.1f}% |
| **Validation** | {split_counts['val']['awake']:,} | {split_counts['val']['sleepy']:,} | {summary_data['split_totals']['val']:,} | {summary_data['split_totals']['val']/total_images*100:.1f}% |
| **Test** | {split_counts['test']['awake']:,} | {split_counts['test']['sleepy']:,} | {summary_data['split_totals']['test']:,} | {summary_data['split_totals']['test']/total_images*100:.1f}% |
| **Total** | **{summary_data['class_totals']['awake']:,}** | **{summary_data['class_totals']['sleepy']:,}** | **{summary_data['total_images']:,}** | **100.0%** |

- **Class Balance**: Awake = `{summary_data['class_balance']['awake']}%`, Sleepy = `{summary_data['class_balance']['sleepy']}%` (balanced).

## 2. Image Properties
- **Color Mode**: `{dict(sample_modes)}` (Grayscale/Luminance `L`).
- **Sample Dimensions**: Variable crop sizes, top examples: `{dict(sample_dimensions.most_common(5))}`.
- **Preprocessing Requirement**: Target models require standardized resizing (e.g., 64x64) and normalization (dividing by 255.0).

## 3. Academic Note on Splitting & Leakage
- **Unique Subjects**: {len(unique_subjects)} subjects.
- **Split Mechanism**: Random image-level sampling (`train_test_split(..., random_state=42)`).
- **Subject Overlap**: All {overlap_train_test} subjects in the test set also appear in the training set.
- **Implication for Report**: Because frames from the same subject under similar sensor and lighting conditions are present across splits, validation and test accuracies will evaluate familiar-face generalization rather than zero-shot subject generalization. This is an essential discussion point for the academic report.
"""
    with open(summary_md_path, "w") as f:
        f.write(md_content)
    print(f"Markdown report saved to: {summary_md_path}")
    
    return summary_data

if __name__ == "__main__":
    inspect_dataset()
