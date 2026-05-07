#!/usr/bin/env python3
"""
Florence-2 Dataset.

Extends BaseVLMDataset with Florence-2 specific formatting.
"""

import sys
from pathlib import Path

# Add project root to path
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from typing import Dict, List, Union, Optional
from training.base import BaseVLMDataset


class Florence2Dataset(BaseVLMDataset):
    """
    Florence-2 format dataset.

    Uses location token format: {category}<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>
    Coordinates normalized to 0-999 range.
    """

    def __init__(
        self,
        annotation_files: Union[str, List[str]],
        dataset_root: str,
        processor,
        max_length: int = 1024,
        task_prompt: str = "<OD>"
    ):
        self.task_prompt = task_prompt
        super().__init__(annotation_files, dataset_root, processor, max_length)

    def _bbox_to_florence_format(self, bbox: List[float], width: int, height: int, label: str) -> str:
        """Convert COCO bbox [x, y, w, h] to Florence-2 location token string."""
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h

        # Normalize to 0-999
        x1_norm = max(0, min(999, int((x1 / width) * 1000)))
        y1_norm = max(0, min(999, int((y1 / height) * 1000)))
        x2_norm = max(0, min(999, int((x2 / width) * 1000)))
        y2_norm = max(0, min(999, int((y2 / height) * 1000)))

        return f"{label}<loc_{x1_norm}><loc_{y1_norm}><loc_{x2_norm}><loc_{y2_norm}>"

    def format_sample(self, sample: Dict) -> Optional[Dict]:
        """Format sample for Florence-2 training (input_ids, attention_mask, pixel_values, labels)."""
        image = self._load_image(sample['image_path'])
        if image is None:
            return None

        # Build ground truth text
        gt_text = ""
        for bbox, label in zip(sample['bboxes'], sample['labels']):
            gt_text += self._bbox_to_florence_format(
                bbox,
                sample['width'],
                sample['height'],
                label
            )

        # Encode input (image + task prompt)
        inputs = self.processor(
            text=self.task_prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        # Encode target (ground truth)
        targets = self.processor.tokenizer(
            gt_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }
