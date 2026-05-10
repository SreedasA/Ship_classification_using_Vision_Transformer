from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from transformers import Trainer, TrainingArguments, ViTForImageClassification, ViTImageProcessor

from ship_vit.config import load_yaml
from ship_vit.data import build_train_dataframe, make_hf_dataset
from ship_vit.transforms import make_collate_fn, make_transforms


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a trained model and save confusion matrix.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--model-dir", type=str, default="runs/altered_vision_transformer")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--out", type=str, default="runs/altered_vision_transformer/confusion_matrix.png")
    args = p.parse_args(argv)

    cfg = load_yaml(args.config)
    tcfg: dict[str, Any] = cfg["train"]

    df = build_train_dataframe(cfg)
    ds, class_labels = make_hf_dataset(df)
    split = ds.train_test_split(
        test_size=float(args.test_size),
        seed=int(tcfg.get("seed", 42)),
        stratify_by_column="label",
    )
    eval_ds = split["test"]

    processor = ViTImageProcessor.from_pretrained(args.model_dir)
    _, val_tfm = make_transforms(processor)

    def _val_transform(batch: dict[str, Any]) -> dict[str, Any]:
        batch["pixel_values"] = [val_tfm(img.convert("RGB")) for img in batch["image"]]
        return batch

    eval_ds = eval_ds.with_transform(_val_transform)

    model = ViTForImageClassification.from_pretrained(args.model_dir)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(Path(args.model_dir) / "_eval_tmp"), report_to="none"),
        eval_dataset=eval_ds,
        data_collator=make_collate_fn(),
        tokenizer=processor,
    )

    outputs = trainer.predict(eval_ds)
    y_true = outputs.label_ids
    y_pred = np.argmax(outputs.predictions, axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels.names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels.names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

