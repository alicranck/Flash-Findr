import os
from platform import processor
from xml.parsers.expat import model
from app.utils.types import ImageHandle, List, NumpyMask, any
from transformers import AutoProcessor, AutoModelForImageTextToText
import numpy as np
import torch
import base64

from .base_tool import BaseVisionTool, ToolKey
from ...utils.image_utils import base64_encode


current_dir = __file__.rsplit("/", 1)[0]


class Captioner(BaseVisionTool): 
    """
    Captioning tool using SmolVLM2.
    """
    def __init__(self, model_id, config, device = 'cpu'):
        self.processor = None
        super().__init__(model_id, config, device)

    def _load_model(self):

        # check if file exists else download to models/
        if not os.path.isfile(self.model_id):
            model_path = os.path.join(f"{current_dir}/../../models", self.model_id)
        else:
            model_path = self.model_id
            
        model = AutoModelForImageTextToText.from_pretrained(model_path,
                                                            torch_dtype=torch.bfloat16,
                                                            _attn_implementation="flash_attention_2")
        self.processor = AutoProcessor.from_pretrained(model_path)

        return model

    def _warmup(self):
        """Implements a dummy captioning run."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        inputs = self.preprocess(dummy_frame)
        _ = self.inference(inputs)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what is happening in this image?"},
                {"type": "image", "data": f"image://base64,{base64_encode(frame, 'png')}"},
                ]
            },
        ]

        inputs = self.processor.apply_chat_template(messages, 
                                                    add_generation_prompt=True,
                                                    tokenize=True,
                                                    return_dict=True,
                                                    return_tensors="pt"
                                                    ).to(self.device, dtype=torch.bfloat16)
        return inputs

    def inference(self, model_inputs: any) -> any:

        generated_ids = self.model.generate(**model_inputs, do_sample=False,
                                             max_new_tokens=64)
        generated_texts = self.processor.batch_decode(generated_ids,
                                                       skip_special_tokens=True)

        return generated_texts

    def postprocess(self, raw_output: any, original_shape: tuple) -> dict:
        data = {"caption": raw_output[0]}
        return data
    
    @property
    def output_keys(self) -> list:
        captions = ToolKey(
            key_name="captions",
            data_type=str,
            description="Generated captions describing the input image",
        )
        return [captions]

    @property
    def processing_input_keys(self) -> list:
        image = ToolKey(
            key_name="image",
            data_type=ImageHandle,
            description="Input image for detection",
            required=True
        )
        return [image]

    def config_keys(self) -> list:
        return []
