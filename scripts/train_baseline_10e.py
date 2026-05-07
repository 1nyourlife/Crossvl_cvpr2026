#!/usr/bin/env python3
"""Florence-2 baseline on MAVREC (mixed ground+aerial, 10 epochs).

Override dataset location with the MAVREC_ROOT env var; defaults assume the
layout described in README.md.
"""

import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from training.florence_2.florence2_trainer import Florence2Trainer

MAVREC_ROOT = os.environ.get("MAVREC_ROOT", "/path/to/MAVREC")
ANNOT_DIR = f"{MAVREC_ROOT}/labelled/supervised_annotations"


def main():
    config = {
        "model_name": "microsoft/Florence-2-base",
        "train_annotation": [
            f"{ANNOT_DIR}/ground/ground_train.json",
            f"{ANNOT_DIR}/aerial/aerial_train.json",
        ],
        "dataset_root": f"{MAVREC_ROOT}/train",
        "output_dir": "checkpoints/mixed_10e_baseline",
        "num_epochs": 10,
        "batch_size": 8,
        "learning_rate": 1e-6,
        "lr_scheduler_type": "cosine",
        "gradient_accumulation_steps": 2,
        "warmup_steps": 500,
        "logging_steps": 50,
        "save_steps": 1000,
        "max_length": 1024,
        "use_lora": False,
    }

    trainer = Florence2Trainer(**config)
    final_model = trainer.train()
    print(f"Training complete. Model saved to: {final_model}")


if __name__ == "__main__":
    main()
