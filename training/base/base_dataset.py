#!/usr/bin/env python3
"""
Base Dataset Class for Vision-Language Models.

Abstract base handling common COCO-format annotation loading logic.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL import Image

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseVLMDataset(Dataset, ABC):
    """
    Abstract base for COCO-format datasets.

    Subclasses must implement format_sample().
    """

    def __init__(
        self,
        annotation_files: Union[str, List[str]],
        dataset_root: str,
        processor,
        max_length: int = 1024
    ):
        if isinstance(annotation_files, str):
            self.annotation_files = [annotation_files]
        else:
            self.annotation_files = annotation_files

        self.dataset_root = Path(dataset_root)
        self.processor = processor
        self.max_length = max_length

        self.samples = self._prepare_training_data()
        logger.info(f"Loaded {len(self.samples)} training samples")

    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None

    def _find_image_path(self, file_name: str) -> Optional[Path]:
        """Try common image directory layouts."""
        possible_paths = [
            self.dataset_root / file_name,
            self.dataset_root / "images" / file_name,
            self.dataset_root / "train" / file_name,
            self.dataset_root / "train" / "img" / file_name,
            self.dataset_root / "test" / "img" / file_name,
            # MAVREC-specific paths
            self.dataset_root / "ground" / file_name,
            self.dataset_root / "aerial" / file_name,
        ]

        for path in possible_paths:
            if path.exists():
                return path

        logger.warning(f"Image not found: {file_name}")
        return None

    def _prepare_training_data(self) -> List[Dict]:
        """Load and merge all annotation files."""
        training_samples = []

        for ann_file in self.annotation_files:
            logger.info(f"Loading annotations from: {ann_file}")
            samples = self._load_coco_annotations(ann_file)
            training_samples.extend(samples)
            logger.info(f"  Loaded {len(samples)} samples from {ann_file}")

        return training_samples

    def _load_coco_annotations(self, ann_file: str, view_type: str = None) -> List[Dict]:
        """
        Parse COCO-format annotation file.

        Args:
            ann_file: annotation file path
            view_type: 'ground' or 'aerial'; inferred from filename if None
        """
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)

        # Auto-detect view type from filename
        if view_type is None:
            if 'ground' in ann_file.lower():
                view_type = 'ground'
            elif 'aerial' in ann_file.lower() or 'drone' in ann_file.lower():
                view_type = 'aerial'
            else:
                view_type = 'unknown'

        # Build lookup indices
        id_to_image = {img['id']: img for img in coco_data['images']}
        id_to_category = {cat['id']: cat['name'] for cat in coco_data['categories']}

        # Group annotations by image
        image_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)

        # Build samples
        samples = []
        for img_id, annotations in image_annotations.items():
            if img_id not in id_to_image:
                continue

            image_info = id_to_image[img_id]

            image_path = self._find_image_path(image_info['file_name'])
            if image_path is None:
                continue

            bboxes = []
            labels = []
            for ann in annotations:
                bbox = ann['bbox']  # COCO format: [x, y, width, height]
                category_id = ann['category_id']
                category_name = id_to_category.get(category_id, 'object')

                bboxes.append(bbox)
                labels.append(category_name)

            sample = {
                'image_path': str(image_path),
                'image_id': img_id,
                'width': image_info['width'],
                'height': image_info['height'],
                'bboxes': bboxes,  # List of [x, y, w, h]
                'labels': labels,  # List of category names
                'view_type': view_type,
            }

            samples.append(sample)

        return samples

    @abstractmethod
    def format_sample(self, sample: Dict) -> Dict:
        """Convert raw sample to model-specific format. Must be implemented by subclasses."""
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return self.format_sample(sample)
