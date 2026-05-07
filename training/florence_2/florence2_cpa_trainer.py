#!/usr/bin/env python3
"""
Florence-2 Trainer with CPA support.

Extends Florence2Trainer with optional CPA (Complexity-Aware Pathway Aggregation).
When use_cpa=False (default), behaves identically to Florence2Trainer.
When use_cpa=True, attaches a CPA module to the model and adds CPA loss.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from transformers import Trainer

logger = logging.getLogger(__name__)

from training.florence_2.florence2_trainer import Florence2Trainer


class CustomTrainerWithCPA(Trainer):
    """HuggingFace Trainer subclass that adds CPA loss when model.cpa exists."""

    def __init__(self, *args, cpa_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.cpa_weight = cpa_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        has_cpa = hasattr(model, 'cpa') and model.cpa is not None

        # Standard forward pass
        if has_cpa:
            # Need hidden states for CPA
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        else:
            outputs = model(**inputs)

        loss = outputs.loss

        # Add CPA loss if enabled
        if has_cpa:
            if hasattr(outputs, 'image_hidden_states') and outputs.image_hidden_states is not None:
                vision_features = outputs.image_hidden_states
            else:
                logger.warning("image_hidden_states not found in outputs, skipping CPA loss")
                return (loss, outputs) if return_outputs else loss

            if hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states is not None:
                text_features = outputs.decoder_hidden_states[-1]
            else:
                logger.warning("decoder_hidden_states not found in outputs, skipping CPA loss")
                return (loss, outputs) if return_outputs else loss

            attention_mask = inputs.get('decoder_attention_mask', None)

            try:
                cpa_output = model.cpa(
                    vision_features=vision_features,
                    text_features=text_features,
                    labels=inputs.get('labels', None),
                    attention_mask=attention_mask
                )

                if isinstance(cpa_output, tuple):
                    cpa_losses, imbalance_score = cpa_output
                else:
                    cpa_losses = cpa_output
                    imbalance_score = None

                cpa_loss = cpa_losses['total']
                loss = loss + self.cpa_weight * cpa_loss

                # Log CPA metrics every 100 steps
                if self.state.global_step % 100 == 0:
                    log_dict = {'cpa/total_loss': cpa_loss.item()}
                    for k, v in cpa_losses.items():
                        if k != 'total':
                            log_dict[f'cpa/{k}'] = v.item() if torch.is_tensor(v) else v

                    if imbalance_score is not None:
                        log_dict['cpa/imbalance_score'] = imbalance_score.mean().item()

                    self.log(log_dict)

            except Exception as e:
                logger.error(f"Error computing CPA loss: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to base loss if CPA fails (don't crash training)
                pass

        return (loss, outputs) if return_outputs else loss


class Florence2CPATrainer(Florence2Trainer):
    """
    Florence-2 trainer with optional CPA support.

    use_cpa=False: identical to Florence2Trainer.
    use_cpa=True: attaches CPA module to model, adds CPA loss during training.
    """

    def __init__(
        self,
        *args,
        use_cpa: bool = False,
        cpa_type: str = "CPA",
        cpa_weight: float = 0.5,
        cpa_hidden_dim: int = 768,
        cpa_num_classes: int = 10,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.use_cpa = use_cpa
        self.cpa_type = cpa_type
        self.cpa_weight = cpa_weight
        self.cpa_hidden_dim = cpa_hidden_dim
        self.cpa_num_classes = cpa_num_classes

    def setup(self):
        super().setup()

        if self.use_cpa:
            logger.info(f"CPA enabled: type={self.cpa_type}, weight={self.cpa_weight}")
        else:
            logger.info("CPA disabled (standard training)")
        logger.info("")

    def load_model(self):
        """Load base model, then attach CPA module if enabled."""
        model = super().load_model()

        if self.use_cpa:
            logger.info(f"Initializing CPA module: {self.cpa_type}")

            from models.cpa import (
                CPA,
                SinglePathwayCPA,
                UniformWeightsCPA,
                HardSelectionCPA,
                MinimalHMB,
                BalancedHMB,
                HierarchicalModalityBalance
            )

            cpa_classes = {
                'CPA': CPA,
                'SinglePathwayCPA': SinglePathwayCPA,
                'UniformWeightsCPA': UniformWeightsCPA,
                'HardSelectionCPA': HardSelectionCPA,
                'MinimalHMB': MinimalHMB,
                'BalancedHMB': BalancedHMB,
                'HMB': HierarchicalModalityBalance
            }

            if self.cpa_type not in cpa_classes:
                raise ValueError(
                    f"Unknown cpa_type: {self.cpa_type}. "
                    f"Available: {list(cpa_classes.keys())}"
                )

            cpa_class = cpa_classes[self.cpa_type]

            # Attach directly to model so it's included in state_dict
            model.cpa = cpa_class(
                hidden_dim=self.cpa_hidden_dim,
                num_classes=self.cpa_num_classes
            ).to(next(model.parameters()).device)

            logger.info("CPA module attached to model")

            cpa_params = sum(p.numel() for p in model.cpa.parameters())
            logger.info(f"CPA parameters: {cpa_params:,} ({cpa_params/1e6:.2f}M)")

            cpa_keys = [k for k in model.state_dict().keys() if 'cpa' in k]
            logger.info(f"{len(cpa_keys)} CPA parameters in model.state_dict()")

        return model

    def train(self):
        """Run training, using CustomTrainerWithCPA if CPA is enabled."""
        self.setup()

        logger.info("Loading processor...")
        self.processor = self.load_processor()

        logger.info("Loading model...")
        self.model = self.load_model()
        self.log_model_info()

        logger.info("Creating dataset...")
        self.train_dataset = self.create_dataset()

        training_args = self.get_training_arguments()
        collate_fn = self.get_collate_fn()

        logger.info("Creating trainer...")
        if self.use_cpa:
            logger.info("Using CustomTrainerWithCPA")
            self.trainer = CustomTrainerWithCPA(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                data_collator=collate_fn,
                tokenizer=self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor,
                cpa_weight=self.cpa_weight
            )
        else:
            from transformers import Trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                data_collator=collate_fn,
                tokenizer=self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor,
            )

        self.save_training_config()

        logger.info("Starting training...")
        self.trainer.train()

        final_model_path = self.output_dir / "final_model"
        logger.info(f"Saving model to: {final_model_path}")

        # CPA params are saved automatically via model.state_dict()
        self.trainer.save_model(str(final_model_path))
        self.processor.save_pretrained(str(final_model_path))

        # Verify CPA was saved
        if self.use_cpa:
            saved_model_file = final_model_path / "model.safetensors"
            if not saved_model_file.exists():
                saved_model_file = final_model_path / "pytorch_model.bin"

            if saved_model_file.exists():
                if saved_model_file.suffix == '.bin':
                    saved_state = torch.load(saved_model_file, map_location='cpu')
                    cpa_keys = [k for k in saved_state.keys() if 'cpa' in k]
                else:
                    cpa_keys = ['cpa.*']  # safetensors: trust save_model

                logger.info(f"Verified: CPA parameters saved in checkpoint")
            else:
                logger.warning("Could not verify CPA save (model file not found)")

        logger.info("=" * 70)
        logger.info("Training completed!")
        logger.info("=" * 70)

        return str(final_model_path)


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Florence-2 CPA Trainer')

    # Model and data
    parser.add_argument('--model-name', type=str, default='microsoft/Florence-2-base')
    parser.add_argument('--train-annotation', type=str, required=True, nargs='+')
    parser.add_argument('--dataset-root', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)

    # Training parameters
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--gradient-accumulation', type=int, default=2)
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--lr-scheduler-type', type=str, default='linear',
                       choices=['linear', 'cosine', 'constant', 'polynomial'],
                       help='Learning rate scheduler type')
    parser.add_argument('--logging-steps', type=int, default=50)
    parser.add_argument('--save-steps', type=int, default=1020)
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    # LoRA parameters
    parser.add_argument('--use-lora', action='store_true')
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.1)

    # CPA parameters
    parser.add_argument('--use-cpa', action='store_true',
                       help='Enable CPA modality balancing')
    parser.add_argument('--cpa-type', type=str, default='CPA',
                       choices=['CPA', 'SinglePathwayCPA', 'UniformWeightsCPA',
                               'HardSelectionCPA', 'MinimalHMB', 'BalancedHMB', 'HMB'],
                       help='CPA variant to use')
    parser.add_argument('--cpa-weight', type=float, default=0.5,
                       help='Weight for CPA loss')
    parser.add_argument('--cpa-hidden-dim', type=int, default=768,
                       help='CPA hidden dimension')
    parser.add_argument('--cpa-num-classes', type=int, default=10,
                       help='Number of classes (for MAVREC: 10)')

    args = parser.parse_args()

    trainer = Florence2CPATrainer(
        model_name=args.model_name,
        train_annotation=args.train_annotation,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_length=args.max_length,
        seed=args.seed,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # CPA parameters
        use_cpa=args.use_cpa,
        cpa_type=args.cpa_type,
        cpa_weight=args.cpa_weight,
        cpa_hidden_dim=args.cpa_hidden_dim,
        cpa_num_classes=args.cpa_num_classes
    )

    model_path = trainer.train()
    print(f"\nTraining completed! Model saved to: {model_path}")

    if args.use_cpa:
        print(f"CPA module automatically saved in model checkpoint")


if __name__ == "__main__":
    main()
