"""
Florence-2 Model Implementation
Microsoft's Florence-2 vision-language model for various vision tasks.
Supports both base models and LoRA fine-tuned models.
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import logging
import numpy as np
from typing import Dict, Any, Optional, Union, List

logger = logging.getLogger(__name__)

# Try to import PEFT for LoRA support
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not available. LoRA models will not be supported.")


class Florence2Model:
    """Florence-2 model wrapper for vision-language tasks."""

    def __init__(self, model_name: str = "microsoft/Florence-2-base", device: Optional[str] = None):
        """
        Initialize Florence-2 model.

        Args:
            model_name: HuggingFace model identifier or local path to model (including LoRA models)
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.is_lora_model = False
        self.base_model_name = "microsoft/Florence-2-base"  # Default base model for LoRA
        self._load_model()

    def _is_local_path(self, path: str) -> bool:
        """Check if the path is a local directory/file."""
        return os.path.exists(path) or "/" in path or "\\" in path

    def _is_lora_model(self, model_path: str) -> bool:
        """Check if the model path contains LoRA adapter files."""
        if not self._is_local_path(model_path):
            return False

        # Check for common LoRA files
        lora_files = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
        return any(os.path.exists(os.path.join(model_path, f)) for f in lora_files)

    def _load_model(self):
        """Load model and processor (supports both base and LoRA models)."""
        try:
            logger.info(f"Loading Florence-2 model: {self.model_name}")

            # Determine if this is a LoRA model
            self.is_lora_model = self._is_lora_model(self.model_name)

            if self.is_lora_model and not PEFT_AVAILABLE:
                raise RuntimeError("LoRA model detected but PEFT is not installed. Please install peft: pip install peft")

            # Load processor
            if self.is_lora_model:
                # For LoRA models, try to load processor from the model directory first
                # If not available, fall back to base model
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    )
                except:
                    logger.info("Processor not found in LoRA model directory, using base model processor")
                    self.processor = AutoProcessor.from_pretrained(
                        self.base_model_name,
                        trust_remote_code=True
                    )
            else:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )

            # Load model
            if self.is_lora_model:
                logger.info("Loading LoRA model...")
                # Load base model first
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    trust_remote_code=True,
                    attn_implementation="eager",
                    low_cpu_mem_usage=True
                )
                # Load LoRA adapter
                self.model = PeftModel.from_pretrained(base_model, self.model_name)
                self.model = self.model.to(self.device)
                logger.info("LoRA model loaded successfully")
            else:
                # Load regular model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    trust_remote_code=True,
                    attn_implementation="eager",  # Avoid SDPA issues
                    low_cpu_mem_usage=True
                ).to(self.device)
                logger.info("Base Florence-2 model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading Florence-2 model: {e}")
            raise

    def predict(self,
                image: Union[str, Image.Image],
                task: str = "<OD>",
                text_input: Optional[str] = None) -> Dict[str, Any]:
        """
        Run inference on image.

        Args:
            image: Input image (PIL Image or path)
            task: Task prompt (e.g., "<OD>", "<CAPTION>", "<DENSE_REGION_CAPTION>")
            text_input: Optional text input for the task

        Returns:
            Dictionary containing prediction results
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        prompt = task
        if text_input:
            prompt = f"{task} {text_input}"

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        # Ensure input tensors match model dtype
        if self.model.dtype == torch.float16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,  # Use beam search for better recall in dense scenes
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=False  # Disable KV cache to avoid the None issue
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height)
        )

        return {
            'task': task,
            'raw_text': generated_text,
            'parsed_result': parsed_answer,
            'image_size': (image.width, image.height)
        }

    def object_detection(self, image: Union[str, Image.Image]) -> Dict[str, Any]:
        """Object detection task."""
        return self.predict(image, task="<OD>")

    def caption(self, image: Union[str, Image.Image]) -> Dict[str, Any]:
        """Image captioning task."""
        return self.predict(image, task="<CAPTION>")

    def detailed_caption(self, image: Union[str, Image.Image]) -> Dict[str, Any]:
        """Detailed image captioning task."""
        return self.predict(image, task="<DETAILED_CAPTION>")

    def ocr(self, image: Union[str, Image.Image]) -> Dict[str, Any]:
        """OCR task."""
        return self.predict(image, task="<OCR>")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'is_lora_model': self.is_lora_model,
            'base_model_name': self.base_model_name if self.is_lora_model else self.model_name,
            'device': self.device,
            'model_type': 'LoRA Fine-tuned' if self.is_lora_model else 'Base Model'
        }

    def predict_with_dropout(self,
                             image: Union[str, Image.Image],
                             task: str = "<OD>",
                             text_input: Optional[str] = None) -> Dict[str, Any]:
        """
        Run inference with MC Dropout (for Box Stability calculation).

        This keeps dropout layers active during inference to introduce controlled randomness.
        Used for measuring model stability, NOT for normal inference.

        Args:
            image: Input image (PIL Image or path)
            task: Task prompt (e.g., "<OD>")
            text_input: Optional text input for the task

        Returns:
            Dictionary containing prediction results
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        prompt = task
        if text_input:
            prompt = f"{task} {text_input}"

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        # Ensure input tensors match model dtype
        if self.model.dtype == torch.float16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

        # MC Dropout: force dropout on during inference
        original_mode = self.model.training
        self.model.train()

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=1,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=False
            )

        # Restore original mode so subsequent calls aren't affected
        self.model.train(original_mode)

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height)
        )

        return {
            'task': task,
            'raw_text': generated_text,
            'parsed_result': parsed_answer,
            'image_size': (image.width, image.height)
        }

    def unload(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        logger.info("Florence-2 model unloaded")