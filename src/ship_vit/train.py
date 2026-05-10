from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)

from ship_vit.config import load_yaml
from ship_vit.data import build_train_dataframe, make_hf_dataset
from ship_vit.metrics import compute_metrics
from ship_vit.transforms import make_collate_fn, make_transforms


def _split_dataset(ds: Dataset, seed: int = 42, test_size: float = 0.2) -> tuple[Dataset, Dataset]:
    split = ds.train_test_split(test_size=test_size, seed=seed, stratify_by_column="label")
    return split["train"], split["test"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train a ViT ship classifier (Hugging Face Trainer).")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--test-size", type=float, default=0.2)
    args = p.parse_args(argv)

    cfg = load_yaml(args.config)
    tcfg: dict[str, Any] = cfg["train"]

    out_dir = Path(tcfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_train_dataframe(cfg)
    ds, class_labels = make_hf_dataset(df)
    train_ds, eval_ds = _split_dataset(ds, seed=int(tcfg.get("seed", 42)), test_size=float(args.test_size))

    model_str = cfg["model"]["pretrained"]
    processor = ViTImageProcessor.from_pretrained(model_str)
    train_tfm, val_tfm = make_transforms(processor)

    def _train_transform(batch: dict[str, Any]) -> dict[str, Any]:
        batch["pixel_values"] = [train_tfm(img.convert("RGB")) for img in batch["image"]]
        return batch

    def _val_transform(batch: dict[str, Any]) -> dict[str, Any]:
        batch["pixel_values"] = [val_tfm(img.convert("RGB")) for img in batch["image"]]
        return batch

    train_ds = train_ds.with_transform(_train_transform)
    eval_ds = eval_ds.with_transform(_val_transform)

    id2label = {i: name for i, name in enumerate(class_labels.names)}
    label2id = {name: i for i, name in id2label.items()}

    model = ViTForImageClassification.from_pretrained(
        model_str,
        num_labels=len(class_labels.names),
        id2label=id2label,
        label2id=label2id,
    )

    train_args = TrainingArguments(
        output_dir=str(out_dir),
        seed=int(tcfg.get("seed", 42)),
        learning_rate=float(tcfg["learning_rate"]),
        per_device_train_batch_size=int(tcfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(tcfg["per_device_eval_batch_size"]),
        num_train_epochs=float(tcfg["num_train_epochs"]),
        weight_decay=float(tcfg["weight_decay"]),
        warmup_steps=int(tcfg.get("warmup_steps", 0)),
        remove_unused_columns=False,
        evaluation_strategy=str(tcfg.get("evaluation_strategy", "epoch")),
        save_strategy=str(tcfg.get("save_strategy", "epoch")),
        save_total_limit=int(tcfg.get("save_total_limit", 2)),
        load_best_model_at_end=bool(tcfg.get("load_best_model_at_end", True)),
        metric_for_best_model=str(tcfg.get("metric_for_best_model", "accuracy")),
        logging_dir=str(tcfg.get("logging_dir", "logs")),
        report_to=str(tcfg.get("report_to", "tensorboard")),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 1)),
        fp16=bool(tcfg.get("fp16", False)),
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=make_collate_fn(),
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    trainer.train()
    metrics = trainer.evaluate()
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))

    (out_dir / "label_map.json").write_text(
        json.dumps({"id2label": id2label, "label2id": label2id}, indent=2),
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

