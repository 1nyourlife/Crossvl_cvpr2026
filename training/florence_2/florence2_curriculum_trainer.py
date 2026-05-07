#!/usr/bin/env python3
"""
Florence-2 Curriculum Trainer with Progressive CPA Weight

Extends Florence2CPATrainer with curriculum learning support:
1. Dynamic CPA weight schedule aligned with curriculum stages
2. Curriculum sampler integration for progressive ground-aerial ratios
3. Enhanced logging for curriculum stages

Design Philosophy (Paired Curriculum):
- Stage 1 (Epoch 0-3): Paired mode + CPA weight = 0.0 (focus on detection)
- Stage 2 (Epoch 4-6): Mixed mode + CPA weight = 0.05 (light alignment)
- Stage 3 (Epoch 7-9): Random mode + CPA weight = 0.1 (full alignment)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from transformers import TrainingArguments

logger = logging.getLogger(__name__)

from training.florence_2.florence2_cpa_trainer import Florence2CPATrainer, CustomTrainerWithCPA


class CustomTrainerWithProgressiveCPA(CustomTrainerWithCPA):
    """
    Custom Trainer with Progressive CPA Weight

    Extends CustomTrainerWithCPA to support dynamic CPA weight
    that changes based on curriculum stage
    """

    def __init__(
        self,
        *args,
        cpa_weight: float = 0.5,
        cpa_weight_schedule: Dict[Tuple[int, int], float] = None,
        **kwargs
    ):
        """
        Initialize trainer with progressive CPA weight

        Args:
            cpa_weight: Default CPA weight (used if schedule not provided)
            cpa_weight_schedule: Schedule mapping (start_epoch, end_epoch) -> weight
                Example: {(0, 3): 0.0, (4, 6): 0.05, (7, 9): 0.1}  # paired curriculum
            **kwargs: Arguments for Trainer
        """
        # Initialize parent with default weight
        super().__init__(*args, cpa_weight=cpa_weight, **kwargs)

        # Store schedule
        if cpa_weight_schedule is None:
            # Default 3-stage schedule aligned with paired curriculum
            self.cpa_weight_schedule = {
                (0, 3): 0.0,    # Stage 1: Paired mode + No CPA (focus on detection)
                (4, 6): 0.05,   # Stage 2: Mixed mode + Light CPA (introduce alignment)
                (7, 9): 0.1,    # Stage 3: Random mode + Full CPA (complete alignment)
            }
        else:
            self.cpa_weight_schedule = cpa_weight_schedule

        # Log schedule
        logger.info(f"Progressive CPA Weight Schedule:")
        for (start, end), weight in sorted(self.cpa_weight_schedule.items()):
            logger.info(f"  Epoch {start}-{end}: weight = {weight}")

        # Current weight (will be updated by get_cpa_weight)
        self.current_cpa_weight = cpa_weight

    def get_cpa_weight(self, epoch: int) -> float:
        """
        Get CPA weight for current epoch

        Args:
            epoch: Current epoch number (0-indexed)

        Returns:
            float: CPA weight for this epoch
        """
        for (start, end), weight in self.cpa_weight_schedule.items():
            if start <= epoch <= end:
                return weight

        # If not found in schedule, use default weight
        logger.warning(f"No CPA weight found for epoch {epoch}, using default {self.cpa_weight}")
        return self.cpa_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with dynamic CPA weight

        Overrides parent's compute_loss to use epoch-dependent CPA weight
        """
        # Get current epoch
        current_epoch = int(self.state.epoch) if hasattr(self.state, 'epoch') and self.state.epoch is not None else 0

        # Update CPA weight based on current epoch
        new_weight = self.get_cpa_weight(current_epoch)

        # Log weight changes
        if new_weight != self.current_cpa_weight:
            logger.info(f"\n{'='*70}")
            logger.info(f"CPA Weight Update: Epoch {current_epoch}")
            logger.info(f"{'='*70}")
            logger.info(f"  Previous weight: {self.current_cpa_weight}")
            logger.info(f"  New weight:      {new_weight}")
            logger.info(f"{'='*70}\n")
            self.current_cpa_weight = new_weight

        # Temporarily override cpa_weight for this computation
        original_weight = self.cpa_weight
        self.cpa_weight = new_weight

        # Call parent's compute_loss
        result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        # Restore original weight
        self.cpa_weight = original_weight

        return result


class Florence2CurriculumTrainer(Florence2CPATrainer):
    """
    Florence-2 Trainer with Curriculum Learning + Progressive CPA

    Combines:
    1. Curriculum sampler (ground->aerial progressive ratios)
    2. Progressive CPA weight (0 -> 0.05 -> 0.1)
    3. Enhanced logging and monitoring

    Usage:
        trainer = Florence2CurriculumTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processor,
            use_cpa=True,
            cpa_weight=0.1,  # max weight
            use_progressive_cpa_weight=True,
            cpa_weight_schedule={(0,3): 0.0, (4,6): 0.05, (7,9): 0.1}  # paired curriculum
        )
    """

    def __init__(
        self,
        *args,
        use_progressive_cpa_weight: bool = False,
        cpa_weight_schedule: Dict[Tuple[int, int], float] = None,
        curriculum_stage_boundaries: List[int] = None,
        **kwargs
    ):
        """
        Initialize curriculum trainer

        Args:
            use_progressive_cpa_weight: Enable progressive CPA weight
            cpa_weight_schedule: Custom CPA weight schedule
            curriculum_stage_boundaries: Epoch boundaries for curriculum stages
                Example: [0, 4, 7, 10] means stages at 0-3, 4-6, 7-9 (paired curriculum)
            **kwargs: Arguments for Florence2CPATrainer
        """
        # Store curriculum config
        self.use_progressive_cpa_weight = use_progressive_cpa_weight
        self.cpa_weight_schedule = cpa_weight_schedule
        self.curriculum_stage_boundaries = curriculum_stage_boundaries or [0, 4, 7, 10]

        # Initialize parent
        super().__init__(*args, **kwargs)

        logger.info(f"\n{'='*70}")
        logger.info(f"Florence2CurriculumTrainer Initialized")
        logger.info(f"{'='*70}")
        logger.info(f"  Use CPA: {self.use_cpa}")
        logger.info(f"  Use Progressive CPA Weight: {use_progressive_cpa_weight}")
        if use_progressive_cpa_weight:
            schedule = cpa_weight_schedule or {(0, 3): 0.0, (4, 6): 0.05, (7, 9): 0.1}
            logger.info(f"  CPA Weight Schedule:")
            for (start, end), weight in sorted(schedule.items()):
                logger.info(f"    Epoch {start}-{end}: {weight}")
        logger.info(f"  Curriculum Stage Boundaries: {self.curriculum_stage_boundaries}")
        logger.info(f"{'='*70}\n")

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Create optimizer and scheduler

        Override to use progressive CPA weight if enabled
        """
        if self.use_cpa and self.use_progressive_cpa_weight:
            logger.info("Using Progressive CPA Weight Training")

        return super().create_optimizer_and_scheduler(num_training_steps)

    def _get_trainer_class(self):
        """
        Get the appropriate Trainer class

        Returns CustomTrainerWithProgressiveCPA if progressive weight enabled
        """
        if self.use_cpa and self.use_progressive_cpa_weight:
            return CustomTrainerWithProgressiveCPA
        else:
            return CustomTrainerWithCPA

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        """
        Main training loop

        Adds curriculum-specific logging
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Starting Curriculum Training")
        logger.info(f"{'='*70}")
        logger.info(f"  Total epochs: {self.args.num_train_epochs}")
        logger.info(f"  Batch size: {self.args.per_device_train_batch_size}")
        logger.info(f"  Learning rate: {self.args.learning_rate}")
        logger.info(f"  CPA weight: {self.cpa_weight} (max)")
        logger.info(f"  Progressive CPA: {self.use_progressive_cpa_weight}")
        logger.info(f"{'='*70}\n")

        # Call parent's train method
        result = super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs
        )

        logger.info(f"\n{'='*70}")
        logger.info(f"Curriculum Training Completed")
        logger.info(f"{'='*70}\n")

        return result

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """
        Override to add curriculum stage logging
        """
        # Log current curriculum stage
        if hasattr(self, 'current_epoch'):
            stage = self._get_curriculum_stage(self.current_epoch)
            if stage is not None:
                logger.debug(f"Current Curriculum Stage: {stage}")

        return super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

    def _get_curriculum_stage(self, epoch: int) -> Optional[int]:
        """
        Get curriculum stage for given epoch

        Args:
            epoch: Current epoch

        Returns:
            int: Stage number (0-indexed), or None if not found
        """
        for i in range(len(self.curriculum_stage_boundaries) - 1):
            start = self.curriculum_stage_boundaries[i]
            end = self.curriculum_stage_boundaries[i + 1] - 1
            if start <= epoch <= end:
                return i
        return None

    def get_train_dataloader(self):
        """
        Get training dataloader

        If curriculum sampler is attached to dataset, set epoch at each call
        """
        dataloader = super().get_train_dataloader()

        # Check if curriculum sampler exists
        if hasattr(dataloader, 'curriculum_sampler'):
            logger.info("Curriculum sampler detected in dataloader")
            self.curriculum_sampler = dataloader.curriculum_sampler
        else:
            self.curriculum_sampler = None

        return dataloader

    def _inner_training_loop(self, *args, **kwargs):
        """
        Override inner training loop to set curriculum sampler epoch
        """
        # Store reference to current epoch
        self.current_epoch = 0

        return super()._inner_training_loop(*args, **kwargs)

    def training_step(self, model, inputs):
        """
        Perform a training step

        Updates curriculum sampler epoch if needed
        """
        # Update curriculum sampler epoch if it exists
        if hasattr(self, 'curriculum_sampler') and self.curriculum_sampler is not None:
            current_epoch = int(self.state.epoch) if hasattr(self.state, 'epoch') and self.state.epoch is not None else 0
            if not hasattr(self, '_last_set_epoch') or self._last_set_epoch != current_epoch:
                self.curriculum_sampler.set_epoch(current_epoch)
                self._last_set_epoch = current_epoch
                self.current_epoch = current_epoch

        return super().training_step(model, inputs)


def create_curriculum_trainer(
    model,
    training_args: TrainingArguments,
    ground_train_dataset,
    aerial_train_dataset,
    eval_dataset,
    tokenizer,
    use_cpa: bool = False,
    cpa_type: str = "CPA",
    cpa_weight: float = 0.1,
    use_progressive_cpa_weight: bool = False,
    cpa_weight_schedule: Dict[Tuple[int, int], float] = None,
    **trainer_kwargs
) -> Florence2CurriculumTrainer:
    """
    Factory function to create curriculum trainer with all components

    Args:
        model: Florence-2 model
        training_args: Training arguments
        ground_train_dataset: Ground view training dataset
        aerial_train_dataset: Aerial view training dataset
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer/processor
        use_cpa: Enable CPA
        cpa_type: CPA architecture type
        cpa_weight: Maximum CPA weight
        use_progressive_cpa_weight: Enable progressive CPA weight
        cpa_weight_schedule: Custom CPA weight schedule
        **trainer_kwargs: Additional trainer arguments

    Returns:
        Florence2CurriculumTrainer instance
    """
    from training.base.curriculum_sampler import create_curriculum_dataloader

    # Create curriculum dataloader
    train_dataloader = create_curriculum_dataloader(
        ground_dataset=ground_train_dataset,
        aerial_dataset=aerial_train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        total_epochs=int(training_args.num_train_epochs),
        num_workers=training_args.dataloader_num_workers,
        shuffle=True,
        drop_last=training_args.dataloader_drop_last,
    )

    # Create trainer
    trainer = Florence2CurriculumTrainer(
        model=model,
        args=training_args,
        train_dataset=None,  # We'll use custom dataloader
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        use_cpa=use_cpa,
        cpa_type=cpa_type,
        cpa_weight=cpa_weight,
        use_progressive_cpa_weight=use_progressive_cpa_weight,
        cpa_weight_schedule=cpa_weight_schedule,
        **trainer_kwargs
    )

    # Attach custom dataloader
    trainer._custom_train_dataloader = train_dataloader

    # Override get_train_dataloader
    def custom_get_train_dataloader():
        return trainer._custom_train_dataloader

    trainer.get_train_dataloader = custom_get_train_dataloader

    logger.info("Curriculum trainer created successfully")

    return trainer
