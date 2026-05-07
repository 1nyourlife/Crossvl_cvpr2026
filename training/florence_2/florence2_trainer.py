#!/usr/bin/env python3
"""
Florence-2 Trainer.

Supports Full Fine-Tuning and LoRA training modes.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    default_data_collator
)

logger = logging.getLogger(__name__)

from training.base import BaseTrainer
from training.florence_2.florence2_dataset import Florence2Dataset

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class Florence2Trainer(BaseTrainer):
    """
    Florence-2 trainer.

    Two modes:
    1. Full Fine-Tuning: use_lora=False
    2. LoRA Fine-Tuning: use_lora=True
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.use_lora and not PEFT_AVAILABLE:
            raise ImportError(
                "LoRA training requires 'peft' package. "
                "Install it with: pip install peft"
            )

    def load_processor(self):
        return AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

    def load_model(self):
        """Load Florence-2 model (Full FT or LoRA)."""
        # Try Flash Attention 2 first, fallback to eager if not available
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            )
            logger.info("Flash Attention 2 enabled for training speedup")
        except Exception as e:
            logger.warning(f"Flash Attention 2 not available: {e}")
            logger.info("Falling back to eager attention implementation")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager"
            )

        if self.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules="all-linear",
                bias="none"
            )
            model = get_peft_model(model, lora_config)
        else:
            # Full fine-tuning
            for param in model.parameters():
                param.requires_grad = True

        return model

    def create_dataset(self):
        return Florence2Dataset(
            annotation_files=self.train_annotation,
            dataset_root=str(self.dataset_root),
            processor=self.processor,
            max_length=self.max_length,
            task_prompt="<OD>"
        )

    def get_collate_fn(self):
        def collate_fn(batch):
            # Filter out None samples (failed image loads)
            batch = [item for item in batch if item is not None]
            if not batch:
                return None
            return default_data_collator(batch)

        return collate_fn


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Florence-2 Trainer (Full FT / LoRA)')

    # Model and data
    parser.add_argument('--model-name', type=str, default='microsoft/Florence-2-base',
                        help='Model name or checkpoint path')
    parser.add_argument('--train-annotation', type=str, required=True, nargs='+',
                        help='Training annotation file(s)')
    parser.add_argument('--dataset-root', type=str, required=True,
                        help='Dataset root directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')

    # Training parameters
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--gradient-accumulation', type=int, default=2,
                        help='Gradient accumulation steps')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='Warmup steps')
    parser.add_argument('--logging-steps', type=int, default=50,
                        help='Logging steps')
    parser.add_argument('--save-steps', type=int, default=1020,
                        help='Save checkpoint steps')
    parser.add_argument('--max-length', type=int, default=1024,
                        help='Max sequence length')

    # LoRA parameters
    parser.add_argument('--use-lora', action='store_true',
                        help='Use LoRA for efficient fine-tuning')
    parser.add_argument('--lora-rank', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.1,
                        help='LoRA dropout')

    args = parser.parse_args()

    trainer = Florence2Trainer(
        model_name=args.model_name,
        train_annotation=args.train_annotation,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_length=args.max_length,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    model_path = trainer.train()
    print(f"\nTraining completed! Model saved to: {model_path}")


if __name__ == "__main__":
    main()
