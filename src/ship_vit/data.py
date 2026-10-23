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
            "label": df[category_col].map(label_map),
        }
    )

    missing = out["label"].isna().sum()
    if missing:
