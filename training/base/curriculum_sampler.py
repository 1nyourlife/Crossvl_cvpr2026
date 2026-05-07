#!/usr/bin/env python3
"""
Paired-based Curriculum Sampler for Cross-View Learning

Implements curriculum learning strategy that leverages MAVREC's unique
paired ground-aerial data to progressively learn cross-view correspondence.

Design Philosophy (NEW):
- Stage 1: Paired training - use synchronized ground-aerial pairs
- Stage 2: Mixed training - combine paired and random sampling
- Stage 3: Random training - baseline-like random sampling

This approach utilizes MAVREC's unique advantage: 100% paired ground-aerial
samples from synchronized cameras at the same location.

Key Insight: Learning from paired views first helps establish cross-view
correspondence before generalizing to random sampling.

References:
- Bengio et al., "Curriculum Learning", ICML 2009
- Dutta et al., "MAVREC: Multiview Aerial Visual Recognition", CVPR 2024
"""

import random
import logging
from typing import Iterator, List, Tuple, Dict
from pathlib import Path
import torch
from torch.utils.data import Sampler, Dataset

logger = logging.getLogger(__name__)


class CurriculumSampler(Sampler):
    """
    Paired-based Curriculum Sampler for Cross-View Learning

    NEW 3-Stage Schedule (10 epochs):
    - Stage 1 (Epoch 0-3): Paired sampling (utilize synchronized ground-aerial pairs)
    - Stage 2 (Epoch 4-6): Mixed sampling (50% paired, 50% random)
    - Stage 3 (Epoch 7-9): Random sampling (baseline-like, test generalization)

    Pairing modes:
    - 'paired': Each batch contains synchronized ground-aerial pairs
    - 'mixed': Mix of paired and random batches
    - 'random': Standard random sampling (ignores pairing)
    """

    def __init__(
        self,
        dataset: Dataset,
        total_epochs: int = 10,
        curriculum_schedule: List[Tuple[Tuple[int, int], str]] = None,
        shuffle: bool = True,
        seed: int = 42
    ):
        """
        Initialize paired-based curriculum sampler

        Args:
            dataset: Combined dataset with view_type field in samples
            total_epochs: Total number of training epochs
            curriculum_schedule: Custom schedule in format:
                [((start_epoch, end_epoch), pairing_mode), ...]
                pairing_mode: 'paired', 'mixed', or 'random'
                If None, uses default 3-stage paired curriculum
            shuffle: Whether to shuffle samples within each epoch
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.shuffle = shuffle
        self.seed = seed

        # Separate ground and aerial indices
        logger.info("Separating ground and aerial samples...")
        self.ground_indices = []
        self.aerial_indices = []
        for idx, sample in enumerate(dataset.samples):
            if sample.get('view_type') == 'ground':
                self.ground_indices.append(idx)
            elif sample.get('view_type') == 'aerial':
                self.aerial_indices.append(idx)

        self.num_ground = len(self.ground_indices)
        self.num_aerial = len(self.aerial_indices)

        logger.info(f"  Ground samples: {self.num_ground}")
        logger.info(f"  Aerial samples: {self.num_aerial}")

        # Current epoch (set by trainer)
        self.current_epoch = 0

        # Default 3-stage paired curriculum schedule
        if curriculum_schedule is None:
            self.schedule = [
                # (epoch_range, pairing_mode)
                ((0, 3), 'paired'),   # Stage 1: Paired learning
                ((4, 6), 'mixed'),    # Stage 2: Mixed (50% paired, 50% random)
                ((7, 9), 'random'),   # Stage 3: Random (baseline-like)
            ]
        else:
            self.schedule = curriculum_schedule

        # Validate schedule
        self._validate_schedule()

        # Build pairing index (scene, frameID) -> (ground_idx, aerial_idx)
        logger.info("Building ground-aerial pairing index...")
        self.pairs = self._build_pairing_index()
        logger.info(f"Found {len(self.pairs)} paired samples (100% paired)")

        # Current pairing mode (will be updated by set_epoch)
        self.pairing_mode = 'paired'

        logger.info(f"Paired Curriculum Sampler initialized:")
        logger.info(f"  Ground dataset: {self.num_ground} samples")
        logger.info(f"  Aerial dataset: {self.num_aerial} samples")
        logger.info(f"  Paired samples: {len(self.pairs)}")
        logger.info(f"  Paired Curriculum schedule:")
        for (start, end), mode in self.schedule:
            logger.info(f"    Epoch {start}-{end}: {mode} mode")
        logger.info("")

    def _build_pairing_index(self) -> List[Tuple[int, int]]:
        """
        Build pairing index matching ground and aerial samples

        MAVREC format:
        - Ground: scene_X_..._groundView_..._XXXXXX.PNG
        - Aerial: scene_X_..._droneView_..._XXXXXX.PNG

        Matching by: (scene, frameID) extracted from filename

        Returns:
            List of (ground_dataset_idx, aerial_dataset_idx) tuples
        """
        import re

        # Extract (scene, frameID) from filename
        def extract_metadata(filename):
            # Pattern: scene_X_..._XXXXXX.PNG
            # Example: scene_12_sdu_30Sec_groundView_6_000810.PNG
            match = re.match(r'scene_(\d+)_.*_(\d{6})\.PNG', filename)
            if match:
                scene = int(match.group(1))
                frame_id = int(match.group(2))
                return (scene, frame_id)
            return None

        # Build ground lookup using ground_indices
        ground_lookup = {}
        for ground_idx in self.ground_indices:
            sample = self.dataset.samples[ground_idx]
            filename = Path(sample['image_path']).name
            metadata = extract_metadata(filename)
            if metadata:
                ground_lookup[metadata] = ground_idx

        # Match with aerial
        pairs = []
        for aerial_idx in self.aerial_indices:
            sample = self.dataset.samples[aerial_idx]
            filename = Path(sample['image_path']).name
            metadata = extract_metadata(filename)
            if metadata and metadata in ground_lookup:
                ground_idx = ground_lookup[metadata]
                pairs.append((ground_idx, aerial_idx))

        if len(pairs) == 0:
            logger.warning("No paired samples found! Dataset may not have pairing metadata.")
            logger.warning("   Falling back to index-based pairing")
            # Fallback: pair by position
            pairs = [(self.ground_indices[i], self.aerial_indices[i])
                     for i in range(min(len(self.ground_indices), len(self.aerial_indices)))]

        return pairs

    def _validate_schedule(self):
        """Validate curriculum schedule"""
        valid_modes = {'paired', 'mixed', 'random'}

        # Check coverage
        covered_epochs = set()
        for (start, end), mode in self.schedule:
            # Validate mode
            if mode not in valid_modes:
                raise ValueError(
                    f"Invalid pairing mode '{mode}' for epoch {start}-{end}. "
                    f"Must be one of: {valid_modes}"
                )

            # Check epoch coverage
            for e in range(start, end + 1):
                if e in covered_epochs:
                    raise ValueError(f"Epoch {e} is covered by multiple schedule entries")
                covered_epochs.add(e)

        expected_epochs = set(range(self.total_epochs))
        if covered_epochs != expected_epochs:
            missing = expected_epochs - covered_epochs
            extra = covered_epochs - expected_epochs
            raise ValueError(
                f"Schedule does not cover all epochs correctly.\n"
                f"Missing epochs: {sorted(missing)}\n"
                f"Extra epochs: {sorted(extra)}"
            )

        logger.info("Curriculum schedule validation passed")

    def set_epoch(self, epoch: int):
        """
        Set current epoch and update pairing mode

        Called by trainer at the beginning of each epoch

        Args:
            epoch: Current epoch number (0-indexed)
        """
        self.current_epoch = epoch

        # Find current stage and pairing mode
        found = False
        for (start, end), mode in self.schedule:
            if start <= epoch <= end:
                self.pairing_mode = mode
                found = True

                # Log stage transition
                if epoch == start:
                    logger.info(f"\n{'='*70}")
                    logger.info(f"Curriculum Stage Change: Epoch {epoch}")
                    logger.info(f"{'='*70}")
                    logger.info(f"  Pairing mode: {mode}")
                    logger.info(f"{'='*70}\n")
                else:
                    logger.debug(f"Epoch {epoch}: Pairing mode = {mode}")

                break

        if not found:
            raise ValueError(f"No schedule entry found for epoch {epoch}")

        # Update random seed for this epoch
        random.seed(self.seed + epoch)

    def __iter__(self) -> Iterator[int]:
        """
        Generate sample indices for current epoch based on pairing mode

        Returns:
            Iterator of integer indices for the combined dataset

        Sampling modes:
        - 'paired': Use synchronized ground-aerial pairs (100% paired)
        - 'mixed': 50% paired sampling + 50% random sampling
        - 'random': Standard random sampling (baseline-like)
        """
        samples = []

        if self.pairing_mode == 'paired':
            # Stage 1: Paired sampling
            # Use all available pairs, shuffle the pairs themselves
            paired_samples = self.pairs.copy()
            if self.shuffle:
                random.shuffle(paired_samples)

            # Flatten pairs to sample list (alternate ground and aerial)
            for ground_idx, aerial_idx in paired_samples:
                samples.append(ground_idx)  # Ground sample (actual dataset index)
                samples.append(aerial_idx)  # Aerial sample (actual dataset index)

            logger.info(f"Epoch {self.current_epoch}: Paired mode - "
                        f"{len(paired_samples)} pairs = {len(samples)} samples "
                        f"(50% ground, 50% aerial)")

        elif self.pairing_mode == 'mixed':
            # Stage 2: Mixed sampling (50% paired + 50% random)
            num_pairs = len(self.pairs)
            num_paired_samples = num_pairs // 2  # Half of pairs used as pairs

            # 1. Paired portion
            paired_samples = self.pairs.copy()
            if self.shuffle:
                random.shuffle(paired_samples)
            paired_portion = paired_samples[:num_paired_samples]

            paired_ground_set = set()
            paired_aerial_set = set()
            for ground_idx, aerial_idx in paired_portion:
                samples.append(ground_idx)
                samples.append(aerial_idx)
                paired_ground_set.add(ground_idx)
                paired_aerial_set.add(aerial_idx)

            # 2. Random portion (use remaining unpaired samples)
            remaining_ground = [idx for idx in self.ground_indices if idx not in paired_ground_set]
            remaining_aerial = [idx for idx in self.aerial_indices if idx not in paired_aerial_set]

            if self.shuffle:
                random.shuffle(remaining_ground)
                random.shuffle(remaining_aerial)

            # Add equal amounts from each view
            num_random_per_view = min(len(remaining_ground), len(remaining_aerial))
            samples.extend(remaining_ground[:num_random_per_view])
            samples.extend(remaining_aerial[:num_random_per_view])

            # Final shuffle to mix paired and random samples
            if self.shuffle:
                random.shuffle(samples)

            num_ground = sum(1 for idx in samples if idx in self.ground_indices)
            num_aerial = len(samples) - num_ground
            logger.info(f"Epoch {self.current_epoch}: Mixed mode - "
                        f"{len(samples)} samples ({num_ground} ground, {num_aerial} aerial)")

        else:  # 'random' mode
            # Stage 3: Random sampling (baseline-like)
            # Independently sample from both views
            all_ground = self.ground_indices.copy()
            all_aerial = self.aerial_indices.copy()

            if self.shuffle:
                random.shuffle(all_ground)
                random.shuffle(all_aerial)

            # Use all samples from both views
            samples.extend(all_ground)
            samples.extend(all_aerial)

            # Final shuffle to mix ground and aerial
            if self.shuffle:
                random.shuffle(samples)

            logger.info(f"Epoch {self.current_epoch}: Random mode - "
                        f"{len(samples)} samples ({self.num_ground} ground, {self.num_aerial} aerial)")

        return iter(samples)

    def __len__(self) -> int:
        """
        Total number of samples per epoch (varies by curriculum stage)

        Returns number of samples based on current pairing mode:
        - 'paired': len(pairs) * 2 (each pair = 2 samples)
        - 'mixed': ~len(pairs) + remaining samples
        - 'random': num_ground + num_aerial
        """
        if self.pairing_mode == 'paired':
            # Each pair contributes 2 samples (ground + aerial)
            return len(self.pairs) * 2

        elif self.pairing_mode == 'mixed':
            # Half paired + remaining random
            num_paired_samples = len(self.pairs) // 2
            paired_count = num_paired_samples * 2  # Each pair = 2 samples

            # Remaining samples (approximate)
            remaining_per_view = (self.num_ground - num_paired_samples)
            random_count = remaining_per_view * 2

            return paired_count + random_count

        else:  # 'random'
            # All samples from both views
            return self.num_ground + self.num_aerial


class CurriculumBatchSampler(torch.utils.data.BatchSampler):
    """
    Batch sampler wrapper for curriculum learning

    Groups samples from CurriculumSampler into batches
    """

    def __init__(
        self,
        curriculum_sampler: CurriculumSampler,
        batch_size: int,
        drop_last: bool = False
    ):
        """
        Initialize curriculum batch sampler

        Args:
            curriculum_sampler: CurriculumSampler instance
            batch_size: Batch size
            drop_last: Whether to drop last incomplete batch
        """
        # Note: We pass curriculum_sampler directly, not wrapped
        # BatchSampler will call __iter__ on it
        super().__init__(
            sampler=curriculum_sampler,
            batch_size=batch_size,
            drop_last=drop_last
        )
        self.curriculum_sampler = curriculum_sampler

    def set_epoch(self, epoch: int):
        """Forward set_epoch to curriculum sampler"""
        self.curriculum_sampler.set_epoch(epoch)


# Utility function to create curriculum dataloader
def create_curriculum_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    total_epochs: int = 10,
    curriculum_schedule: List[Tuple[Tuple[int, int], str]] = None,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = False,
    seed: int = 42,
    **dataloader_kwargs
):
    """
    Create a DataLoader with paired-based curriculum sampling

    Args:
        dataset: Combined dataset with view_type field in samples
        batch_size: Batch size
        total_epochs: Total training epochs
        curriculum_schedule: Custom curriculum schedule in format:
            [((start_epoch, end_epoch), pairing_mode), ...]
            pairing_mode: 'paired', 'mixed', or 'random'
            If None, uses default 3-stage paired curriculum
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle samples
        drop_last: Whether to drop last incomplete batch
        seed: Random seed
        **dataloader_kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader with paired-based curriculum sampling
    """
    from torch.utils.data import DataLoader

    # Create curriculum sampler
    curriculum_sampler = CurriculumSampler(
        dataset=dataset,
        total_epochs=total_epochs,
        curriculum_schedule=curriculum_schedule,
        shuffle=shuffle,
        seed=seed
    )

    # Create batch sampler
    batch_sampler = CurriculumBatchSampler(
        curriculum_sampler=curriculum_sampler,
        batch_size=batch_size,
        drop_last=drop_last
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        **dataloader_kwargs
    )

    # Attach sampler for epoch setting
    dataloader.curriculum_sampler = curriculum_sampler

    return dataloader
