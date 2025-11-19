import os
from app.utils.types import ImageHandle, List, NumpyMask, Any
from ultralytics import YOLOE  # type: ignore
from ultralytics.engine.results import Boxes  # type: ignore
import numpy as np

from .base_tool import BaseVisionTool, ToolKey


current_dir = __file__.rsplit("/", 1)[0]
DEFAULT_IMAGE_SIZE = 640
DEFAULT_CONFIDENCE_THRESHOLD = 0.25


class OpenVocabularyDetector(BaseVisionTool):
    """
    Detection tool using an open-vocabulary YOLO model.
    Used for unconstrained zero-shot object detection based on a custom vocabulary.
    """
    def __init__(self, model_id, config, device = 'cpu'):
        self.imgsz: int
        self.conf_threshold: float
        self.vocabulary: List[str] | None
        super().__init__(model_id, config, device)

    def _configure(self, config: dict):
        self.imgsz = config.get('imgsz', DEFAULT_IMAGE_SIZE)
        self.conf_threshold = config.get('conf_threshold', DEFAULT_CONFIDENCE_THRESHOLD)
        self.vocabulary = config.get('vocabulary', None)
        return

    def _load_model(self):

        if self.vocabulary is None:
            raise ValueError("OpenVocabularyDetector requires a 'vocabulary' list in the config.")
        
        # check if file exists else download to models/
        if not os.path.isfile(self.model_id):
            model_path = os.path.join(f"{current_dir}/../../models", self.model_id)
            model = YOLOE(model_path)
        else:
            model = YOLOE(self.model_id)

        pos_embeddings = model.get_text_pe(self.vocabulary)
        model.set_classes(self.vocabulary, pos_embeddings)
        onnx_model = self.compile_onnx_model(model, imgsz=self.imgsz)

        return onnx_model

    def _warmup(self):
        """Implements a dummy YOLO predict run."""
        for i in range(16):
            dummy_frame = np.zeros((640, 480, 3), dtype=np.uint8)
            self.model.predict(dummy_frame)

    def set_vocabulary(self, classes: list):
        """Custom method for this tool."""
        if self.model:
            self.model.set_classes(classes)
            print(f"INFO: DetectionTool vocabulary set to: {classes}")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        # Note: YOLOv8's .predict() handles pre-processing,
        return frame 

    def inference(self, model_inputs: np.ndarray) -> Any:
        """Runs YOLO inference."""
        return self.model.track(model_inputs,
                                 conf=self.conf_threshold,
                                  imgsz=self.imgsz,
                                   tracker='bytetrack.yaml')

    def postprocess(self, raw_output: Any, original_shape: tuple) -> dict:
        """Parses YOLO results and updates the data dict."""
        results = {raw_output[0]} # Get the first result
        data = {
            "boxes": results.boxes,
            "masks": np.array(results.masks) if results.masks is not None else None,
            "keypoints": results.keypoints
        }
        return data

    @staticmethod
    def compile_onnx_model(model, imgsz=640):
        onnx_model = model.export(format="onnx", dynamic=True,
                                   imgsz=imgsz, batch=1)
        ort_session = YOLOE(onnx_model)
        return ort_session
    
    @property
    def output_keys(self) -> list:
        boxes = ToolKey(
            key_name="boxes",
            data_type=Boxes,
            description="Detected bounding boxes",
        )
        masks = ToolKey(
            key_name="masks",
            data_type=NumpyMask,
            description="Detected masks (if applicable)",
        )
        keypoints = ToolKey(
            key_name="keypoints",
            data_type=any,
            description="Detected keypoints (if applicable)",
        )
        return [boxes, masks, keypoints]
    
    @property
    def processing_input_keys(self) -> list:
        image = ToolKey(
            key_name="image",
            data_type=ImageHandle,
            description="Input image for detection",
            required=True
        )
        return [image]

    @property
    def config_keys(self) -> list:
        vocabulary = ToolKey(
            key_name="vocabulary",
            data_type=List[str],
            description="List of classes for open-vocabulary detection",
            required=True
        )
        image_size = ToolKey(
            key_name="image_size",
            data_type=int,
            description="Image size for model input (default: 640)",
        )
        confidence = ToolKey(
            key_name="confidence_threshold",
            data_type=float,
            description="Confidence threshold for detections (default: 0.25)",
        )
        return [vocabulary, image_size, confidence]
