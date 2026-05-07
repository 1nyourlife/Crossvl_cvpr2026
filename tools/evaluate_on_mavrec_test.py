#!/usr/bin/env python3
"""
Evaluate Florence-2 models on MAVREC_TEST dataset
Handles scattered image files and computes COCO mAP
"""
import sys
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import warnings
from PIL import Image

# Suppress known warnings from supervision NMS
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='supervision')

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.florence2 import Florence2Model
from tools.mavrec_test_image_finder import MAVRECTestImageFinder
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    import supervision as sv
    HAS_SUPERVISION = True
except ImportError:
    HAS_SUPERVISION = False
    print("Warning: supervision not available, NMS will be skipped")


def run_inference_with_nms(model, image_path, nms_threshold=0.5):
    """Run inference and apply NMS"""
    result = model.predict(image_path, task="<OD>")

    if '<OD>' not in result['parsed_result']:
        return [], []

    od_result = result['parsed_result']['<OD>']
    labels = od_result.get('labels', [])
    bboxes = od_result.get('bboxes', [])

    if len(labels) == 0:
        return [], []

    # Apply NMS if supervision available
    if HAS_SUPERVISION and len(bboxes) > 0:
        xyxy = np.array(bboxes)
        confidence = np.ones(len(bboxes))
        class_id = np.array([hash(l) % 1000 for l in labels])

        sv_det = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        sv_det = sv_det.with_nms(threshold=nms_threshold)

        # Extract filtered results
        labels = [labels[i] for i in range(len(labels)) if i < len(sv_det)]
        bboxes = sv_det.xyxy.tolist()

    return labels, bboxes


def evaluate_on_test_set(checkpoint, test_annotation, test_root, output_file, dataset_type='ground', max_images=0):
    """Evaluate model on MAVREC_TEST"""
    print(f"\n{'='*70}")
    print(f"Evaluating on MAVREC_TEST - {dataset_type.upper()}")
    print(f"{'='*70}\n")

    # Check if using simple structure (MAVREC_AERIAL_TEST) or complex structure (MAVREC_TEST)
    simple_aerial_dir = os.path.join(test_root, "aerial_test")
    use_simple_structure = os.path.exists(simple_aerial_dir) and dataset_type == 'aerial'

    if use_simple_structure:
        print(f"Using simple directory structure: {simple_aerial_dir}")
        image_finder = None
    else:
        # Initialize image finder for complex structure
        image_finder = MAVRECTestImageFinder(test_root)
        stats = image_finder.get_stats()
        print(f"Image cache stats: {stats['total']} images ({stats['ground']} ground, {stats['aerial']} aerial)")

    # Load annotations
    print(f"\nLoading annotations: {test_annotation}")

    # Fix COCO format - add missing fields if needed
    with open(test_annotation, 'r') as f:
        ann_data = json.load(f)

    # Add missing COCO fields if not present
    if 'info' not in ann_data:
        ann_data['info'] = {
            'description': 'MAVREC TEST Dataset',
            'version': '1.0',
            'year': 2024
        }
    if 'licenses' not in ann_data:
        ann_data['licenses'] = []

    # Save to temporary file with complete COCO format
    import tempfile
    temp_ann = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(ann_data, temp_ann)
    temp_ann.close()

    coco_gt = COCO(temp_ann.name)
    image_ids = coco_gt.getImgIds()

    # Clean up temp file after loading
    os.unlink(temp_ann.name)

    # Limit number of images if specified
    if max_images > 0 and max_images < len(image_ids):
        image_ids = image_ids[:max_images]
        print(f"Limited to first {max_images} images for quick validation")

    # Build category mapping
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    cat_name_to_id = {cat['name']: cat['id'] for cat in categories}

    print(f"\nDataset info:")
    print(f"  Images: {len(image_ids)}")
    print(f"  Categories: {len(categories)}")
    print(f"  NMS: {'enabled' if HAS_SUPERVISION else 'disabled'}")

    # Load model
    print(f"\nLoading model from: {checkpoint}")
    model = Florence2Model(checkpoint, device='cuda')
    print("Model loaded\n")

    # Run inference
    print("Running inference on test set...")
    results = []
    not_found = []
    invalid_labels = []

    for img_id in tqdm(image_ids, desc="Processing images"):
        img_info = coco_gt.loadImgs(img_id)[0]
        file_name = img_info['file_name']

        # Find image path
        if use_simple_structure:
            # Simple structure: images directly in aerial_test/
            image_path = os.path.join(simple_aerial_dir, file_name)
        else:
            # Complex structure: use image finder
            image_path = image_finder.find(file_name)

        if image_path is None or not os.path.exists(image_path):
            not_found.append(file_name)
            continue

        # Inference
        try:
            labels, bboxes = run_inference_with_nms(model, image_path)

            # Get actual image dimensions and annotation dimensions
            actual_image = Image.open(image_path)
            actual_width, actual_height = actual_image.size

            # Get annotation dimensions
            ann_width = img_info['width']
            ann_height = img_info['height']

            # Calculate scaling factors
            scale_x = ann_width / actual_width
            scale_y = ann_height / actual_height

            # Convert to COCO format with coordinate scaling
            for label, bbox in zip(labels, bboxes):
                if label not in cat_name_to_id:
                    invalid_labels.append((file_name, label))
                    continue

                x1, y1, x2, y2 = bbox

                # Scale from actual image coordinates to annotation coordinates
                scaled_x1 = x1 * scale_x
                scaled_y1 = y1 * scale_y
                scaled_w = (x2 - x1) * scale_x
                scaled_h = (y2 - y1) * scale_y

                results.append({
                    'image_id': img_id,
                    'category_id': cat_name_to_id[label],
                    'bbox': [scaled_x1, scaled_y1, scaled_w, scaled_h],  # COCO format: [x, y, w, h]
                    'score': 1.0
                })
        except Exception as e:
            print(f"\nError processing {file_name}: {e}")
            continue

    print(f"\nInference complete!")
    print(f"  Generated {len(results)} detections")
    print(f"  Images not found: {len(not_found)}")
    print(f"  Invalid labels: {len(invalid_labels)}")

    if not_found:
        print(f"\nSample missing files:")
        for fname in not_found[:5]:
            print(f"    - {fname}")

    if invalid_labels:
        print(f"\nSample invalid labels:")
        from collections import Counter
        label_counts = Counter([label for _, label in invalid_labels])
        for label, count in label_counts.most_common(5):
            print(f"    - '{label}': {count} times")

    # Evaluate
    print(f"\nComputing COCO metrics...")
    if len(results) == 0:
        print("No valid detections generated!")
        metrics = {
            'mAP': 0.0,
            'mAP50': 0.0,
            'mAP75': 0.0,
            'mAP_small': 0.0,
            'mAP_medium': 0.0,
            'mAP_large': 0.0,
        }
    else:
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        metrics = {
            'mAP': float(coco_eval.stats[0]),
            'mAP50': float(coco_eval.stats[1]),
            'mAP75': float(coco_eval.stats[2]),
            'mAP_small': float(coco_eval.stats[3]),
            'mAP_medium': float(coco_eval.stats[4]),
            'mAP_large': float(coco_eval.stats[5]),
        }

    # Save results
    output = {
        'checkpoint': checkpoint,
        'test_annotation': test_annotation,
        'dataset_type': dataset_type,
        'metrics': metrics,
        'num_predictions': len(results),
        'num_images': len(image_ids),
        'num_not_found': len(not_found),
        'num_invalid_labels': len(invalid_labels)
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"\n{'='*70}")
    print(f"TEST RESULTS - {dataset_type.upper()}")
    print(f"{'='*70}")
    print(f"mAP:    {metrics['mAP']*100:.2f}%")
    print(f"mAP50:  {metrics['mAP50']*100:.2f}%")
    print(f"mAP75:  {metrics['mAP75']*100:.2f}%")
    print(f"{'='*70}\n")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate on MAVREC_TEST')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--test-annotation', required=True, help='Test annotation JSON file')
    parser.add_argument('--test-root', required=True,
                        help='MAVREC_TEST root directory')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--dataset-type', choices=['ground', 'aerial'], default='ground',
                        help='Dataset type (ground or aerial)')
    parser.add_argument('--max-images', type=int, default=0,
                        help='Maximum number of images to evaluate (0 = all images)')

    args = parser.parse_args()

    evaluate_on_test_set(
        checkpoint=args.checkpoint,
        test_annotation=args.test_annotation,
        test_root=args.test_root,
        output_file=args.output,
        dataset_type=args.dataset_type,
        max_images=args.max_images
    )
