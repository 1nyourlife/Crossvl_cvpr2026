#!/bin/bash
# Train Florence-2 with Paired Curriculum Learning (no CPA), 10 epochs.
# Curriculum: epochs 0-3 paired, 4-6 mixed, 7-9 random.

set -e

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR="checkpoints/curriculum_only_10e_${TIMESTAMP}"
mkdir -p "$CHECKPOINT_DIR"

MAVREC_ROOT="${MAVREC_ROOT:-/path/to/MAVREC}"
GROUND_TRAIN="$MAVREC_ROOT/labelled/supervised_annotations/ground/ground_train.json"
AERIAL_TRAIN="$MAVREC_ROOT/labelled/supervised_annotations/aerial/aerial_train.json"
GROUND_VAL="$MAVREC_ROOT/labelled/supervised_annotations/ground/ground_val.json"
AERIAL_VAL="$MAVREC_ROOT/labelled/supervised_annotations/aerial/aerial_val.json"
TRAIN_IMAGE_ROOT="$MAVREC_ROOT/train"
VAL_IMAGE_ROOT="$MAVREC_ROOT/val"

MODEL_NAME="microsoft/Florence-2-base"

NUM_EPOCHS=10
BATCH_SIZE=8
LEARNING_RATE=1e-6
WARMUP_STEPS=1076
MAX_LENGTH=1024
GRADIENT_ACCUMULATION_STEPS=2

USE_CPA=false
USE_PROGRESSIVE_CPA_WEIGHT=false

cat > "$CHECKPOINT_DIR/training_config.json" <<EOF
{
  "model_name": "$MODEL_NAME",
  "ground_train_annotation": "$GROUND_TRAIN",
  "aerial_train_annotation": "$AERIAL_TRAIN",
  "ground_val_annotation": "$GROUND_VAL",
  "aerial_val_annotation": "$AERIAL_VAL",
  "train_image_root": "$TRAIN_IMAGE_ROOT",
  "val_image_root": "$VAL_IMAGE_ROOT",
  "num_epochs": $NUM_EPOCHS,
  "batch_size": $BATCH_SIZE,
  "learning_rate": $LEARNING_RATE,
  "warmup_steps": $WARMUP_STEPS,
  "max_length": $MAX_LENGTH,
  "gradient_accumulation_steps": $GRADIENT_ACCUMULATION_STEPS,
  "use_cpa": $USE_CPA,
  "use_progressive_cpa_weight": $USE_PROGRESSIVE_CPA_WEIGHT,
  "curriculum_type": "paired-based",
  "curriculum_schedule": [
    {"epochs": "0-3", "mode": "paired"},
    {"epochs": "4-6", "mode": "mixed"},
    {"epochs": "7-9", "mode": "random"}
  ],
  "start_time": "$(date -Iseconds)"
}
EOF

python -u <<EOF
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('$CHECKPOINT_DIR/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

project_root = Path('$PROJECT_ROOT')
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
from training.florence_2.florence2_dataset import Florence2Dataset
from training.base.curriculum_sampler import create_curriculum_dataloader

processor = AutoProcessor.from_pretrained("$MODEL_NAME", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "$MODEL_NAME",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

train_dataset = Florence2Dataset(
    annotation_files=['$GROUND_TRAIN', '$AERIAL_TRAIN'],
    dataset_root='$TRAIN_IMAGE_ROOT',
    processor=processor,
    max_length=$MAX_LENGTH,
)

training_args = TrainingArguments(
    output_dir='$CHECKPOINT_DIR',
    num_train_epochs=$NUM_EPOCHS,
    per_device_train_batch_size=$BATCH_SIZE,
    gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS,
    learning_rate=$LEARNING_RATE,
    warmup_steps=$WARMUP_STEPS,
    logging_steps=100,
    save_strategy='epoch',
    save_total_limit=3,
    eval_strategy='no',
    remove_unused_columns=False,
    dataloader_num_workers=0,
    bf16=True,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    optim="adamw_torch_fused",
    report_to='none',
)

train_dataloader = create_curriculum_dataloader(
    dataset=train_dataset,
    batch_size=$BATCH_SIZE,
    total_epochs=$NUM_EPOCHS,
    num_workers=4,
    shuffle=True,
    drop_last=False,
)


class CurriculumEpochCallback(TrainerCallback):
    def __init__(self, curriculum_sampler):
        self.curriculum_sampler = curriculum_sampler

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else 0
        self.curriculum_sampler.set_epoch(epoch)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=None,
    eval_dataset=None,
    tokenizer=processor,
    callbacks=[CurriculumEpochCallback(train_dataloader.curriculum_sampler)],
)

trainer._custom_train_dataloader = train_dataloader
trainer.curriculum_sampler = train_dataloader.curriculum_sampler
trainer.get_train_dataloader = lambda: trainer._custom_train_dataloader

try:
    trainer.train()
    trainer.save_model('$CHECKPOINT_DIR/final_model')
    processor.save_pretrained('$CHECKPOINT_DIR/final_model')
    logger.info(f"Model saved to $CHECKPOINT_DIR/final_model")
except Exception as e:
    logger.error(f"Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

EOF

echo "Done. Checkpoint at: $CHECKPOINT_DIR"
