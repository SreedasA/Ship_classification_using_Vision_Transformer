from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from datasets import ClassLabel, Dataset, Image


def build_train_dataframe(cfg: dict[str, Any]) -> pd.DataFrame:
    dcfg = cfg["dataset"]
    root = Path(dcfg["root"])
    train_csv = root / dcfg["train_csv"]
    train_images_dir = root / dcfg["train_images_dir"]

    df = pd.read_csv(train_csv)
    image_col = dcfg.get("image_column_in_csv", "image")
    category_col = dcfg.get("category_column_in_csv", "category")

    label_map_raw = dcfg["label_map"]
    label_map = {int(k): str(v) for k, v in label_map_raw.items()}

    if image_col not in df.columns:
        raise KeyError(f"Missing column '{image_col}' in {train_csv}")
    if category_col not in df.columns:
        raise KeyError(f"Missing column '{category_col}' in {train_csv}")

    out = pd.DataFrame(
        {
            "image": (train_images_dir / df[image_col].astype(str)).astype(str),
            "label": df[category_col].map(label_map).astype(str),
        }
    )

    missing = out["label"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} rows have unmapped labels. Check 'label_map' in config."
        )
    return out


def make_hf_dataset(df: pd.DataFrame) -> tuple[Dataset, ClassLabel]:
    labels_list = sorted(df["label"].unique().tolist())
    class_labels = ClassLabel(num_classes=len(labels_list), names=labels_list)

    ds = Dataset.from_pandas(df)
    ds = ds.cast_column("image", Image())

    def _map_label2id(example: dict[str, Any]) -> dict[str, Any]:
        example["label"] = class_labels.str2int(example["label"])
        return example

    ds = ds.map(_map_label2id)
    ds = ds.cast_column("label", class_labels)
    return ds, class_labels

