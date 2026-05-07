#!/usr/bin/env python3
"""
Base Trainer Class for Vision-Language Models.

Abstract base class defining the common training pipeline and interface.
"""

import os
import sys
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

import torch
from transformers import (
    TrainingArguments,
    Trainer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base for all trainers.

    Subclasses must implement:
    - load_model()
    - load_processor()
    - create_dataset()
    - get_collate_fn()
    """

    def __init__(
        self,
        model_name: str,
        train_annotation: Union[str, List[str]],
        dataset_root: str,
        output_dir: str,
        # Training hyperparameters
        num_epochs: int = 10,
        batch_size: int = 2,
        learning_rate: float = 1e-5,
        gradient_accumulation_steps: int = 2,
        warmup_steps: int = 1000,
        lr_scheduler_type: str = "linear",
        logging_steps: int = 50,
        save_steps: int = 1020,
        max_length: int = 1024,
        seed: int = 42,
        # LoRA parameters (optional, only for LoRA-based trainers)
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        **kwargs
    ):
        self.model_name = model_name
        self.train_annotation = [train_annotation] if isinstance(train_annotation, str) else train_annotation
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)

        # Training hyperparameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.lr_scheduler_type = lr_scheduler_type
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_length = max_length
        self.seed = seed

        # LoRA parameters
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # Additional kwargs
        self.kwargs = kwargs

        # To be initialized
        self.model = None
        self.processor = None
        self.train_dataset = None
        self.trainer = None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup(self):
        """Log training configuration."""
        logger.info("=" * 70)
        logger.info(f"{self.__class__.__name__} Setup")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Training data: {self.train_annotation}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Epochs: {self.num_epochs}, Batch: {self.batch_size}, LR: {self.learning_rate}")
        logger.info(f"Max length: {self.max_length}")
        if self.use_lora:
            logger.info(f"LoRA: rank={self.lora_rank}, alpha={self.lora_alpha}, dropout={self.lora_dropout}")
        else:
            logger.info("Training mode: Full fine-tuning")
        logger.info("")

    @abstractmethod
    def load_model(self):
        """Load model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def load_processor(self):
        """Load processor/tokenizer. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def create_dataset(self):
        """Create training dataset. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_collate_fn(self):
        """Return data collation function. Must be implemented by subclasses."""
        pass

    def get_training_arguments(self) -> TrainingArguments:
        """Build TrainingArguments. Can be overridden by subclasses."""
        return TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            lr_scheduler_type=self.lr_scheduler_type,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=3,
            bf16=True,
            gradient_checkpointing=True,  # Critical for Full FT stability
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
            max_grad_norm=1.0,
            optim="adamw_torch_fused",  # Faster optimizer
            seed=self.seed
        )

    def log_model_info(self):
        """Log trainable vs total parameter counts."""
        if self.model is not None:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def save_training_config(self):
        """Dump training config to JSON."""
        config = {
            "model_name": self.model_name,
            "train_annotation": self.train_annotation,
            "dataset_root": str(self.dataset_root),
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_length": self.max_length,
            "use_lora": self.use_lora,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "start_time": datetime.now().isoformat(),
        }

        if self.use_lora:
            config.update({
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
            })

        config_file = self.output_dir / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Training config saved to: {config_file}")

    def train(self):
        """
        Full training pipeline.

        Returns:
            str: path to saved model
        """
        # 1. Setup
        self.setup()

        # 2. Load processor
        logger.info("Loading processor...")
        self.processor = self.load_processor()

        # 3. Load model
        logger.info("Loading model...")
        self.model = self.load_model()
        self.log_model_info()

        # 4. Create dataset
        logger.info("Creating dataset...")
        self.train_dataset = self.create_dataset()

        # 5. Get training arguments
        training_args = self.get_training_arguments()

        # 6. Get collate function
        collate_fn = self.get_collate_fn()

        # 7. Create trainer
        logger.info("Creating trainer...")
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=collate_fn,
            tokenizer=self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor,
        )

        # 8. Save config
        self.save_training_config()

        # 9. Train
        logger.info("Starting training...")
        self.trainer.train()

        # 10. Save model
        final_model_path = self.output_dir / "final_model"
        logger.info(f"Saving model to: {final_model_path}")
        self.trainer.save_model(str(final_model_path))
        self.processor.save_pretrained(str(final_model_path))

        logger.info("=" * 70)
        logger.info("Training completed!")
        logger.info("=" * 70)

        return str(final_model_path)
