#!/bin/bash
# Train Florence-2 + CPA on MAVREC, 10 epochs.

set -e

MAVREC_ROOT="${MAVREC_ROOT:-/path/to/MAVREC}"
GROUND_TRAIN="${MAVREC_ROOT}/labelled/supervised_annotations/ground/ground_train.json"
AERIAL_TRAIN="${MAVREC_ROOT}/labelled/supervised_annotations/aerial/aerial_train.json"
DATASET_ROOT="${MAVREC_ROOT}/train"

MODEL_NAME="microsoft/Florence-2-base"

NUM_EPOCHS=10
BATCH_SIZE=8
GRADIENT_ACCUMULATION=2
LEARNING_RATE=1e-6
LR_SCHEDULER="cosine"
WARMUP_STEPS=1076
MAX_LENGTH=1024

CPA_TYPE="CPA"
CPA_WEIGHT=0.1
CPA_HIDDEN_DIM=768
CPA_NUM_CLASSES=4

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="checkpoints/full_cpa_10e_${TIMESTAMP}"

for path in "${GROUND_TRAIN}" "${AERIAL_TRAIN}"; do
    if [ ! -f "${path}" ]; then
        echo "Error: annotation not found: ${path}" >&2
        exit 1
    fi
done
if [ ! -d "${DATASET_ROOT}" ]; then
    echo "Error: dataset root not found: ${DATASET_ROOT}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/training_config.json" <<EOF
{
  "model_name": "${MODEL_NAME}",
  "train_annotation": [
    "${GROUND_TRAIN}",
    "${AERIAL_TRAIN}"
  ],
  "dataset_root": "${DATASET_ROOT}",
  "num_epochs": ${NUM_EPOCHS},
  "batch_size": ${BATCH_SIZE},
  "learning_rate": ${LEARNING_RATE},
  "lr_scheduler_type": "${LR_SCHEDULER}",
  "warmup_steps": ${WARMUP_STEPS},
  "max_length": ${MAX_LENGTH},
  "gradient_accumulation_steps": ${GRADIENT_ACCUMULATION},
  "use_cpa": true,
  "cpa_type": "${CPA_TYPE}",
  "cpa_weight": ${CPA_WEIGHT},
  "cpa_hidden_dim": ${CPA_HIDDEN_DIM},
  "cpa_num_classes": ${CPA_NUM_CLASSES},
  "start_time": "$(date -Iseconds)"
}
EOF

python training/florence_2/florence2_cpa_trainer.py \
    --model-name "${MODEL_NAME}" \
    --train-annotation "${GROUND_TRAIN}" "${AERIAL_TRAIN}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-epochs ${NUM_EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --learning-rate ${LEARNING_RATE} \
    --gradient-accumulation ${GRADIENT_ACCUMULATION} \
    --max-length ${MAX_LENGTH} \
    --save-steps 1020 \
    --logging-steps 50 \
    --lr-scheduler-type ${LR_SCHEDULER} \
    --warmup-steps ${WARMUP_STEPS} \
    --use-cpa \
    --cpa-type ${CPA_TYPE} \
    --cpa-weight ${CPA_WEIGHT} \
    --cpa-hidden-dim ${CPA_HIDDEN_DIM} \
    --cpa-num-classes ${CPA_NUM_CLASSES}

echo "Done. Checkpoint at: ${OUTPUT_DIR}"
