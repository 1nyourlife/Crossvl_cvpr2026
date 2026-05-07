#!/usr/bin/env python3
"""
Generate qualitative comparison visualizations for CVPR supplementary material.
Uses PIL only (no matplotlib) to avoid dependency issues.
"""
import sys
import os
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.florence2 import Florence2Model

try:
    import supervision as sv
    HAS_SUPERVISION = True
except ImportError:
    HAS_SUPERVISION = False
    print("Warning: supervision not available, NMS disabled")


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

    # Apply NMS
    if HAS_SUPERVISION and len(bboxes) > 0:
        xyxy = np.array(bboxes)
        confidence = np.ones(len(bboxes))
        class_id = np.array([hash(l) % 1000 for l in labels])

        sv_det = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        sv_det = sv_det.with_nms(threshold=nms_threshold)

        labels = [labels[i] for i in range(len(labels)) if i < len(sv_det)]
        bboxes = sv_det.xyxy.tolist()

    return labels, bboxes


def draw_detections_pil(image, labels, bboxes):
    """Draw bounding boxes on PIL image"""
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Color map
    color_map = {
        'person': (255, 0, 0),
        'car': (0, 255, 0),
        'bicycle': (0, 0, 255),
        'motorcycle': (255, 255, 0),
    }

    for label, bbox in zip(labels, bboxes):
        x1, y1, x2, y2 = bbox

        # Validate bbox coordinates
        if x2 <= x1 or y2 <= y1:
            continue  # Skip invalid bbox

        color = color_map.get(label, (128, 128, 128))

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label
        text_bbox = draw.textbbox((x1, y1-25), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1-25), label, fill='white', font=font)

    return draw_image


def add_title(image, title, num_detections):
    """Add title bar to top of image"""
    width, height = image.size
    title_height = 60

    # Create new image with title bar
    new_image = Image.new('RGB', (width, height + title_height), 'white')

    # Draw title
    draw = ImageDraw.Draw(new_image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    text = f"{title}\n({num_detections} detections)"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_x = (width - text_width) // 2
    text_y = (title_height - text_height) // 2
    draw.text((text_x, text_y), text, fill='black', font=font)

    # Paste original image below title
    new_image.paste(image, (0, title_height))

    return new_image


def create_comparison_figure(image_path, checkpoints, output_path):
    """Create side-by-side comparison with PIL"""
    # Load original
    original_image = Image.open(image_path).convert('RGB')
    img_width, img_height = original_image.size

    # Run inference
    results = {}
    for method_name, checkpoint_path in checkpoints.items():
        print(f"  Running {method_name}...", end=' ', flush=True)
        model = Florence2Model(checkpoint_path, device='cuda')
        labels, bboxes = run_inference_with_nms(model, image_path)
        results[method_name] = {'labels': labels, 'bboxes': bboxes}
        print(f"({len(bboxes)} objects)")
        del model

    # Create annotated images
    annotated_images = []

    # Original with title
    orig_titled = add_title(original_image, "Original Image", 0)
    annotated_images.append(orig_titled)

    # Method results
    for method_name, result in results.items():
        annotated = draw_detections_pil(original_image, result['labels'], result['bboxes'])
        titled = add_title(annotated, method_name, len(result['bboxes']))
        annotated_images.append(titled)

    # Concatenate horizontally
    total_width = sum(img.width for img in annotated_images)
    max_height = max(img.height for img in annotated_images)

    combined = Image.new('RGB', (total_width, max_height), 'white')

    x_offset = 0
    for img in annotated_images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save
    combined.save(output_path, quality=95)
    print(f"Saved: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Generate qualitative comparisons for supplementary material')
    parser.add_argument('--image-paths', nargs='+', required=True,
                       help='Paths to test images to visualize')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for visualizations')
    parser.add_argument('--baseline-checkpoint', required=True,
                       help='Baseline checkpoint path')
    parser.add_argument('--cpa-checkpoint', required=True,
                       help='CPA checkpoint path')
    parser.add_argument('--curriculum-checkpoint', required=True,
                       help='Curriculum (PCL) checkpoint path')
    parser.add_argument('--combined-checkpoint', required=True,
                       help='CrossVL (CPA + PCL) checkpoint path')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    checkpoints = {
        'Baseline': args.baseline_checkpoint,
        '+ CPA': args.cpa_checkpoint,
        '+ Curriculum': args.curriculum_checkpoint,
        '+ Both': args.combined_checkpoint,
    }

    print(f"\n{'='*70}")
    print("Generating Supplementary Visualizations (PIL version)")
    print(f"{'='*70}\n")

    print("Using checkpoints:")
    for name, path in checkpoints.items():
        print(f"  {name:15s}: {path}")
    print()

    for idx, image_path in enumerate(args.image_paths):
        print(f"[{idx+1}/{len(args.image_paths)}] Processing: {Path(image_path).name}")

        if not os.path.exists(image_path):
            print(f"  ✗ Image not found: {image_path}")
            continue

        output_path = os.path.join(args.output_dir,
                                  f"comparison_{idx+1:02d}_{Path(image_path).stem}.png")

        try:
            create_comparison_figure(image_path, checkpoints, output_path)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"Generated {len(args.image_paths)} comparison figures")
    print(f"Saved to: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
