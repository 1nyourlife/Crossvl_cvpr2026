#!/usr/bin/env python3
"""
COCO-style evaluation script for OWL-ViT and Florence-2 results.

This script evaluates detection results against COCO ground truth annotations
and computes standard metrics like mAP, AP50, AP75, etc.

Usage:
  python tools/eval_coco_map.py \
    --gt-file /path/to/ground_truth.json \
    --pred-dir /path/to/predictions/ \
    --output-dir /path/to/evaluation_results/
"""

import os
import json
import argparse
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging

# Ensure project root is on PYTHONPATH
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("Warning: pycocotools not available. Install with: pip install pycocotools")

logger = logging.getLogger(__name__)


class DetectionEvaluator:
    """Evaluator for detection results."""
    
    def __init__(self, gt_file: str, pred_dir: str, output_dir: str):
        """
        Initialize evaluator.
        
        Args:
            gt_file: Path to COCO ground truth JSON file
            pred_dir: Directory containing prediction JSON files
            output_dir: Directory to save evaluation results
        """
        self.gt_file = gt_file
        self.pred_dir = pred_dir
        self.output_dir = output_dir
        self.coco_gt = None
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load ground truth
        self._load_ground_truth()
    
    def _load_ground_truth(self):
        """Load COCO ground truth annotations."""
        try:
            self.coco_gt = COCO(self.gt_file)
            logger.info(f"Loaded ground truth with {len(self.coco_gt.imgs)} images and {len(self.coco_gt.anns)} annotations")
        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")
            raise
    
    def _load_predictions(self, model_name: str) -> List[Dict]:
        """
        Load predictions from JSON files.
        
        Args:
            model_name: Name of the model (e.g., 'owlvit', 'florence2')
            
        Returns:
            List of prediction dictionaries in COCO format
        """
        predictions = []
        
        # Find all prediction JSON files
        json_files = []
        for root, dirs, files in os.walk(self.pred_dir):
            for file in files:
                if file.endswith('_predictions.json') or file.endswith('_detections.json'):
                    json_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(json_files)} prediction files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Check if data is already in COCO format (list of predictions)
                if isinstance(data, list):
                    predictions.extend(data)
                else:
                    # Extract image info
                    image_id = self._get_image_id_from_filename(json_file)
                    if image_id is None:
                        continue

                    # Convert predictions to COCO format
                    for detection in data.get('detections', []):
                        bbox = detection.get('bbox', [])
                        if len(bbox) != 4:
                            continue

                        # Convert [x1, y1, x2, y2] to [x, y, width, height]
                        x1, y1, x2, y2 = bbox
                        width = x2 - x1
                        height = y2 - y1

                        # Get category ID
                        category_id = self._get_category_id(detection.get('label', 'object'))

                        prediction = {
                            'image_id': image_id,
                            'category_id': category_id,
                            'bbox': [x1, y1, width, height],
                            'score': detection.get('score', 1.0)
                        }
                        predictions.append(prediction)

            except Exception as e:
                logger.warning(f"Error processing {json_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(predictions)} predictions for {model_name}")
        return predictions
    
    def _get_image_id_from_filename(self, filename: str) -> Optional[int]:
        """Extract image ID from filename."""
        try:
            # Extract filename without extension
            basename = os.path.basename(filename)
            name_without_ext = basename.replace('_detections.json', '')
            
            # Find matching image in COCO dataset
            for img_id, img_info in self.coco_gt.imgs.items():
                if img_info['file_name'].replace('.PNG', '').replace('.jpg', '').replace('.jpeg', '') == name_without_ext:
                    return img_id
            
            return None
        except Exception as e:
            logger.warning(f"Error extracting image ID from {filename}: {e}")
            return None
    
    def _get_category_id(self, label: str) -> int:
        """Get category ID for a given label from the ground truth dataset."""
        # Build category name to ID mapping from ground truth
        if not hasattr(self, '_category_name_to_id'):
            self._category_name_to_id = {}
            for cat in self.coco_gt.dataset['categories']:
                self._category_name_to_id[cat['name'].lower()] = cat['id']

        # Semantic mapping for common variations
        semantic_mapping = {
            'automobile': 'car',
            'motor vehicle': 'car',
            'vehicle': 'car',  # Map generic 'vehicle' to 'car' if no 'vehicle' category exists
            'suv': 'car',
            'sedan': 'car',
            'coupe': 'car',
            'hatchback': 'car',
            'pedestrian': 'person',
            'human': 'person',
            'people': 'person',
            'traffic signal': 'traffic light',
            'stoplight': 'traffic light',
            'street light': 'streetlight',
            'lamp post': 'streetlight',
            'trolley': 'tram',
            'train': 'tram',
            'bike': 'bicycle',
            'cycle': 'bicycle',
            'lorry': 'truck',
            'pickup': 'truck',
            'minivan': 'van',
        }

        label_lower = label.lower()

        # Try direct match first
        if label_lower in self._category_name_to_id:
            return self._category_name_to_id[label_lower]

        # Try semantic mapping
        if label_lower in semantic_mapping:
            mapped_label = semantic_mapping[label_lower]
            if mapped_label in self._category_name_to_id:
                return self._category_name_to_id[mapped_label]

        # If no match and there's an 'other' category, use it
        if 'other' in self._category_name_to_id:
            return self._category_name_to_id['other']

        # Last resort: return the first category ID (usually 1)
        if self._category_name_to_id:
            return min(self._category_name_to_id.values())

        return 1  # Fallback
    
    def evaluate_model(self, model_name: str) -> Dict[str, Any]:
        """
        Evaluate a specific model.
        
        Args:
            model_name: Name of the model to evaluate
            
        Returns:
            Dictionary containing evaluation results
        """
        if not PYCOCOTOOLS_AVAILABLE:
            logger.error("pycocotools not available. Cannot perform evaluation.")
            return {}
        
        logger.info(f"Evaluating {model_name}...")
        
        # Load predictions
        predictions = self._load_predictions(model_name)
        
        if not predictions:
            logger.warning(f"No predictions found for {model_name}")
            return {}
        
        # Save predictions in COCO format
        pred_file = os.path.join(self.output_dir, f"{model_name}_predictions.json")
        with open(pred_file, 'w') as f:
            json.dump(predictions, f)

        # Load predictions into COCO format (pass list directly to avoid file parsing issues)
        coco_dt = self.coco_gt.loadRes(predictions)
        
        # Initialize COCO evaluator
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        
        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract results
        results = {
            'model_name': model_name,
            'num_predictions': len(predictions),
            'mAP': coco_eval.stats[0],  # mAP @ IoU=0.50:0.95
            'mAP_50': coco_eval.stats[1],  # mAP @ IoU=0.50
            'mAP_75': coco_eval.stats[2],  # mAP @ IoU=0.75
            'mAP_small': coco_eval.stats[3],  # mAP for small objects
            'mAP_medium': coco_eval.stats[4],  # mAP for medium objects
            'mAP_large': coco_eval.stats[5],  # mAP for large objects
            'AR_1': coco_eval.stats[6],  # AR given 1 detection per image
            'AR_10': coco_eval.stats[7],  # AR given 10 detections per image
            'AR_100': coco_eval.stats[8],  # AR given 100 detections per image
            'AR_small': coco_eval.stats[9],  # AR for small objects
            'AR_medium': coco_eval.stats[10],  # AR for medium objects
            'AR_large': coco_eval.stats[11],  # AR for large objects
        }
        
        # Save detailed results
        results_file = os.path.join(self.output_dir, f"{model_name}_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation completed for {model_name}")
        logger.info(f"mAP: {results['mAP']:.4f}, mAP@50: {results['mAP_50']:.4f}, mAP@75: {results['mAP_75']:.4f}")
        
        return results
    
    def evaluate_all_models(self, model_names: List[str]) -> Dict[str, Any]:
        """
        Evaluate all specified models.
        
        Args:
            model_names: List of model names to evaluate
            
        Returns:
            Dictionary containing all evaluation results
        """
        all_results = {}
        
        for model_name in model_names:
            try:
                results = self.evaluate_model(model_name)
                all_results[model_name] = results
            except Exception as e:
                import traceback
                logger.error(f"Error evaluating {model_name}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                all_results[model_name] = {}
        
        # Save combined results
        combined_file = os.path.join(self.output_dir, "all_results.json")
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Print summary
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        for model_name, model_results in results.items():
            if not model_results:
                print(f"{model_name}: No results available")
                continue
                
            print(f"\n{model_name.upper()}:")
            print(f"  mAP:     {model_results.get('mAP', 0):.4f}")
            print(f"  mAP@50:  {model_results.get('mAP_50', 0):.4f}")
            print(f"  mAP@75:  {model_results.get('mAP_75', 0):.4f}")
            print(f"  AR@1:    {model_results.get('AR_1', 0):.4f}")
            print(f"  AR@10:   {model_results.get('AR_10', 0):.4f}")
            print(f"  Predictions: {model_results.get('num_predictions', 0)}")
        
        print("\n" + "="*80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate OWL-ViT and Florence-2 detection results")
    parser.add_argument("--gt-file", required=True, help="Path to COCO ground truth JSON file")
    parser.add_argument("--pred-dir", required=True, help="Directory containing prediction JSON files")
    parser.add_argument("--output-dir", required=True, help="Directory to save evaluation results")
    parser.add_argument("--models", nargs="+", default=["owlvit", "florence2"], 
                       help="Model names to evaluate")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize evaluator
    evaluator = DetectionEvaluator(args.gt_file, args.pred_dir, args.output_dir)
    
    # Run evaluation
    results = evaluator.evaluate_all_models(args.models)
    
    print(f"\nEvaluation completed. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()













