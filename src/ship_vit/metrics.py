from __future__ import annotations

from typing import Any

import evaluate
import numpy as np
from sklearn.metrics import f1_score


_accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    pred_labels = np.argmax(preds, axis=1)

    acc = _accuracy.compute(predictions=pred_labels, references=labels)["accuracy"]
    f1 = f1_score(labels, pred_labels, average="macro")
    return {"accuracy": float(acc), "f1_macro": float(f1)}

