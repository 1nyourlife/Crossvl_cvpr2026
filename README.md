# CrossVL

PyTorch implementation of *CrossVL: Complexity-Aware Feature Routing and Paired Curriculum for Cross-View Vision-Language Detection* (CVPR 2026).

CrossVL targets cross-view object detection on synchronized aerial-ground image pairs. Two components:

- **Complexity-Aware Pathway Aggregation (CPA)**: dynamic feature routing through three parallel pathways (dense / medium / sparse), weighted by a per-image complexity score.
- **Paired Curriculum Learning (PCL)**: curriculum sampler that progresses from synchronized aerial-ground pairs to mixed/random sampling over training.

## Installation

```bash
git clone https://github.com/1nyourlife/CrossVL.git
cd CrossVL
pip install -r requirements.txt
```

Tested with Python 3.8, PyTorch 2.0+, CUDA 11.8.

## Dataset

Download MAVREC and arrange so that `$MAVREC_ROOT` contains:

```
$MAVREC_ROOT/
├── labelled/supervised_annotations/
│   ├── ground/{ground_train,ground_val}.json
│   └── aerial/{aerial_train,aerial_val}.json
├── train/   # training images
└── val/     # validation images
```

The training scripts read `$MAVREC_ROOT` from the environment.

## Training

```bash
export MAVREC_ROOT=/path/to/MAVREC

# Florence-2 baseline
python scripts/train_baseline_10e.py

# + CPA
bash scripts/train_full_cpa_10e.sh

# + PCL
bash scripts/train_curriculum_only_10e.sh

# CrossVL (CPA + PCL)
bash scripts/train_curriculum_cpa_10e.sh
```

## Evaluation

```bash
python tools/evaluate_on_mavrec_test.py \
    --checkpoint /path/to/checkpoint \
    --test-annotation /path/to/test_annotation.json \
    --test-root /path/to/MAVREC_TEST \
    --output evaluation_results.json
```

## Results

Complete COCO metrics on MAVREC aerial view, 3-seed mean ± std (seeds 42, 123, 789). Checkpoint selected per seed by validation mAP, then averaged.

| Method     | Val mAP          | Val mAP50        | Val mAP75        | Test mAP          | Test mAP50        | Test mAP75        |
|------------|------------------|------------------|------------------|-------------------|-------------------|-------------------|
| Baseline   | 63.73 ± 2.31     | 75.22 ± 1.89     | 68.62 ± 2.45     | 58.66 ± 1.57      | 71.06 ± 2.18      | 64.19 ± 2.34      |
| + CPA      | 64.49 ± 0.89     | 76.00 ± 0.71     | 69.58 ± 0.93     | 60.66 ± 1.09      | 72.61 ± 1.01      | 66.48 ± 1.15      |
| + PCL      | 64.37 ± 5.21     | 76.62 ± 5.82     | 69.63 ± 5.67     | 56.53 ± 4.97      | 68.40 ± 5.34      | 61.88 ± 5.12      |
| **CrossVL**| **65.35 ± 1.28** | **76.79 ± 1.11** | **70.16 ± 1.35** | **61.03 ± 1.50**  | **73.13 ± 1.42**  | **66.91 ± 1.58**  |

CPA notably stabilises training (test-mAP std drops from 1.57 to 1.09 vs baseline; PCL alone shows 4.97). Per-scale (mAPS/mAPM), checkpoint-selection, ablations, and additional analyses are in the CVPR supplementary material.

## Layout

```
models/        CPA module, Florence-2 wrapper
training/      base trainer/dataset, Florence-2 trainers, PCL sampler
scripts/       four entry-point training scripts
tools/         MAVREC evaluator, COCO mAP, supplementary visuals
supplementary/ qualitative comparison figures
```

## Citation

```bibtex
@inproceedings{crossvl2026,
  title     = {CrossVL: Complexity-Aware Feature Routing and Paired Curriculum for Cross-View Vision-Language Detection},
  author    = {Liu, Zhipeng and Luo, Chunbo},
  booktitle = {CVPR},
  year      = {2026}
}
```

## Acknowledgments

Built on [Florence-2](https://huggingface.co/microsoft/Florence-2-base). MAVREC dataset reference: see paper.

## License

MIT.
