import os
from app.utils.types import ImageHandle, Any
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
from optimum.intel.openvino.modeling_visual_language import OVModelForVisualCausalLM
import numpy as np
import torch

from .base_tool import BaseVisionTool, ToolKey
from ...utils.image_utils import base64_encode


current_dir = __file__.rsplit("/", 1)[0]


class Captioner(BaseVisionTool): 
    """
    Captioning tool using SmolVLM2.
    """
    def __init__(self, model_id, config, device = 'cpu'):
        self.processor = None
        self.tokenizer = None
        super().__init__(model_id, config, device)

    def _load_model(self):

        model = OVModelForVisualCausalLM.from_pretrained(self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        return model

    def preprocess(self, frame: np.ndarray) -> np.ndarray:

        messages = [
            {
                "role": "user",
                "content": [
                {"type": "text", "text": "Give a concise description of what is happening in this image"},
                {"type": "image", "url": f"data:image/png;base64,{base64_encode(frame, 'png')}"},
                ]
            },
        ]

        inputs = self.processor.apply_chat_template(messages, 
                                                    add_generation_prompt=True,
                                                    tokenize=True,
                                                    return_dict=True,
                                                    return_tensors="pt"
                                                    ).to(self.device)
        return inputs

    def inference(self, model_inputs: Any) -> Any:

        generated_ids = self.model.generate(**model_inputs, do_sample=False,
                                             max_new_tokens=64)
        generated_texts = self.processor.batch_decode(generated_ids,
                                                       skip_special_tokens=True)

        return generated_texts

    def postprocess(self, raw_output: Any, original_shape: tuple) -> dict:
        assitant_response = raw_output[0].split("Assistant:")[1].strip()
        data = {"caption": assitant_response}
        return data
    
    @property
    def output_keys(self) -> list:
        caption = ToolKey(
            key_name="caption",
            data_type=str,
            description="Generated captions describing the input image",
        )
        return [caption]

    @property
    def processing_input_keys(self) -> list:
        return []

    @property
    def config_keys(self) -> list:
        return []
