# Dataset Inspection Summary: MRL Eye Dataset

- **Dataset Root**: `H:\Code Lab\DriverDrowsiness\dataset`
- **Total Images**: `84,898`
- **Corrupted / Unreadable Images**: `0`

## 1. Split & Class Distribution

| Split | Awake | Sleepy | Total | % of Dataset |
|---|---|---|---|---|
| **Train** | 25,770 | 25,167 | 50,937 | 60.0% |
| **Validation** | 8,591 | 8,389 | 16,980 | 20.0% |
| **Test** | 8,591 | 8,390 | 16,981 | 20.0% |
| **Total** | **42,952** | **41,946** | **84,898** | **100.0%** |

- **Class Balance**: Awake = `50.59%`, Sleepy = `49.41%` (balanced).

## 2. Image Properties
- **Color Mode**: `{'L': 3000}` (Grayscale/Luminance `L`).
- **Sample Dimensions**: Variable crop sizes, top examples: `{'84x84': 171, '82x82': 170, '83x83': 169, '81x81': 141, '86x86': 124}`.
- **Preprocessing Requirement**: Target models require standardized resizing (e.g., 64x64) and normalization (dividing by 255.0).

## 3. Academic Note on Splitting & Leakage
- **Unique Subjects**: 37 subjects.
- **Split Mechanism**: Random image-level sampling (`train_test_split(..., random_state=42)`).
- **Subject Overlap**: All 37 subjects in the test set also appear in the training set.
- **Implication for Report**: Because frames from the same subject under similar sensor and lighting conditions are present across splits, validation and test accuracies will evaluate familiar-face generalization rather than zero-shot subject generalization. This is an essential discussion point for the academic report.
