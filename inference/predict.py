from __future__ import annotations

import argparse

from inference.ensemble import build_ensemble
from inference.pipeline import run_model
from src.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BirdCLEF notebook-equivalent inference")
    parser.add_argument("--model", choices=["model_22", "model_51", "ensemble"], required=True)
    parser.add_argument("--config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model == "ensemble":
        build_ensemble(args.config or "config/ensemble.yaml")
        return
    config = load_config(args.config or f"config/{args.model}.yaml")
    submission = run_model(config, args.model)
    output = config["paths"]["output"]
    submission.to_csv(output, index=False)
    print(f"Wrote {output} with shape {submission.shape}")


if __name__ == "__main__":
    main()
