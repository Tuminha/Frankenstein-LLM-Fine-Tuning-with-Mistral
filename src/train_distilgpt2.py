"""CLI entry for DistilGPT-2 CPU finetune (optional path)."""

import yaml
from pathlib import Path


def main(config_path="configs/train.yaml"):
    """
    Main training function for DistilGPT-2 CPU finetune.
    
    # TODO:
    # 1) Parse YAML config
    # 2) Load dataset (Hub or CSV)
    # 3) Tokenize with DistilGPT-2 tokenizer
    # 4) Train on CPU with TrainingArguments
    # 5) Save model to outputs/
    
    Args:
        config_path: Path to YAML configuration file
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()

