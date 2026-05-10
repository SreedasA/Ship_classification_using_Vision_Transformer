from __future__ import annotations

"""
Minimal inference example (for README / recruiters).

Usage:
  python scripts/inference_example.py --model-dir runs/altered_vision_transformer --image path/to/image.jpg
"""

import argparse

from ship_vit.predict import main as predict_main


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, default="runs/altered_vision_transformer")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    return predict_main(
        [
            "--model-dir",
            args.model_dir,
            "--image",
            args.image,
            "--top-k",
            str(args.top_k),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

