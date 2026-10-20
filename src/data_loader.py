import os
from pathlib import Path
import tensorflow as tf

# Standardized class mapping: 0 -> sleepy (closed eyes), 1 -> awake (open eyes)
CLASS_NAMES = ["sleepy", "awake"]
DEFAULT_IMG_SIZE = (64, 64)
DEFAULT_BATCH_SIZE = 32

def load_dataset(
    dataset_dir: str = "dataset",
    split: str = "train",
    img_size: tuple = DEFAULT_IMG_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    seed: int = 42,
    color_mode: str = "grayscale"
) -> tf.data.Dataset:
    """
    Loads MRL Eye dataset split using tf.keras.utils.image_dataset_from_directory.
    Explicitly maps 'sleepy' -> 0 and 'awake' -> 1.
    """
    split_dir = Path(dataset_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(split_dir),
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=img_size,
        shuffle=shuffle,
        seed=seed if shuffle else None
    )
    
    # Enable autotune prefetching for pipeline performance
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

def get_smoke_subsets(
    dataset_dir: str = "dataset",
    train_samples: int = 256,
    val_samples: int = 64,
    batch_size: int = DEFAULT_BATCH_SIZE,
    img_size: tuple = DEFAULT_IMG_SIZE,
    seed: int = 42
):
    """
    Constructs isolated small training and validation subsets for rapid smoke testing.
    Ensures training subset is used exclusively for training and validation subset
    strictly for evaluation.
    """
    train_batches = max(1, train_samples // batch_size)
    val_batches = max(1, val_samples // batch_size)

    full_train = load_dataset(
        dataset_dir=dataset_dir,
        split="train",
        img_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )
    
    full_val = load_dataset(
        dataset_dir=dataset_dir,
        split="val",
        img_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    smoke_train = full_train.take(train_batches)
    smoke_val = full_val.take(val_batches)

    return smoke_train, smoke_val
