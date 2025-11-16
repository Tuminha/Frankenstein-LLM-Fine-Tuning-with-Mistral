"""CLI entry for QLoRA (used outside notebooks if desired)."""

import yaml
from pathlib import Path


def main(config_path="configs/train.yaml"):
    """
    Main training function for QLoRA.
    
    # TODO:
    # 1) Parse YAML config
    # 2) Load/pull dataset from Hub or CSV
    # 3) Tokenize dataset
    # 4) Load 4-bit model + LoRA config
    # 5) Train + save adapters
    # 6) Optional push to Hub
    
    Args:
        config_path: Path to YAML configuration file
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()

