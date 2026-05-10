from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from transformers import pipeline

from ship_vit.config import load_yaml


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run inference with a trained ViT ship classifier.")
    p.add_argument("--model-dir", type=str, default="runs/altered_vision_transformer")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--config", type=str, default="configs/default.yaml", help="Used only for default top_k.")
    args = p.parse_args(argv)

    if args.top_k is None:
        cfg = load_yaml(args.config)
        args.top_k = int(cfg.get("inference", {}).get("top_k", 3))

    clf = pipeline("image-classification", model=args.model_dir)
    img = Image.open(Path(args.image)).convert("RGB")
    preds = clf(img, top_k=int(args.top_k))

    for p_ in preds:
        print(f"{p_['label']}\t{p_['score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

