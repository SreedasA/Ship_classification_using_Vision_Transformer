from __future__ import annotations

from typing import Any

import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomAdjustSharpness,
    RandomHorizontalFlip,
    RandomRotation,
    Resize,
    ToTensor,
)


def make_transforms(image_processor: Any) -> tuple[Compose, Compose]:
    image_mean, image_std = image_processor.image_mean, image_processor.image_std
    size = image_processor.size["height"]
    normalize = Normalize(mean=image_mean, std=image_std)

    train_tfm = Compose(
        [
            Resize((size, size)),
            RandomRotation(45),
            RandomAdjustSharpness(2),
            RandomHorizontalFlip(0.5),
            ToTensor(),
            normalize,
        ]
    )

    val_tfm = Compose(
        [
            Resize((size, size)),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    return train_tfm, val_tfm


def make_collate_fn() -> Any:
    def collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        labels = torch.tensor([ex["label"] for ex in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    return collate_fn

