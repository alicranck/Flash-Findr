import os
from collections import defaultdict
from app.utils.types import ImageHandle, List, NumpyMask, Any
from trackers import SORTTracker
import supervision as sv
from ultralytics import YOLOE  # type: ignore
from ultralytics.engine.results import Boxes  # type: ignore
import numpy as np

from .base_tool import BaseVisionTool, ToolKey
from ...utils.image_utils import load_image_pil


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
        self.extrapolated_frames: int
        self.tracker: SORTTracker
        self.tracking_history: Dict[str, List]
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

        self.tracker = SORTTracker(lost_track_buffer=5, frame_rate=10, 
                                    minimum_consecutive_frames=2, minimum_iou_threshold=0.2)
        self.tracking_history = defaultdict(list)

        return onnx_model

    def set_vocabulary(self, classes: list):
        if self.model:
            self.model.set_classes(classes)
            print(f"INFO: DetectionTool vocabulary set to: {classes}")

    def inference(self, model_inputs: np.ndarray) -> Any:
        """Runs YOLO inference."""
        results = self.model.predict(model_inputs,
                                    conf=self.conf_threshold,
                                    imgsz=self.imgsz)
        detections = sv.Detections.from_ultralytics(results[0])
        detections = self.tracker.update(detections)

        self.extrapolated_frames = 0

        for i in range(len(detections)):

            xyxy_box = detections.xyxy[i]
            track_id = detections.tracker_id[i]
            confidence = detections.confidence[i]
            class_id = detections.class_id[i]

            self.tracking_history[track_id].append({"xyxy": xyxy_box,
                                                    "conf": confidence,
                                                    "cls": class_id,
                                                    "id": track_id})

        finished_tracks = self.tracking_history.keys() - set(detections.tracker_id)
        for ft_id in finished_tracks:
            ft = self.tracking_history.pop(ft_id)

        return results

    def postprocess(self, raw_output: Any, original_shape: tuple) -> dict:
        """Parses YOLO results and updates the data dict."""
        results = raw_output[0]
        data = {
            "boxes": results.boxes,
            "masks": results.masks,
            "keypoints": results.keypoints,
            "class_names": results.names
        }
        return data
    
    def extrapolate_last(self, frame_handle: ImageHandle) -> Any:
        extrapolated_boxes = []
        self.extrapolated_frames += 1
        for track_id, boxes in self.tracking_history.items():
            extrapolated_boxes.append({"xyxy": self.extrapolate_box(boxes) ,
                                        "conf": boxes[-1]["conf"],
                                        "cls": boxes[-1]["cls"],
                                        "id": track_id})

        results = self.postprocess(self.last_result, None)
        if len(extrapolated_boxes) > 0:
            results["boxes"] = extrapolated_boxes

        return results

    def extrapolate_box(self, boxes: list) -> list:

        if self.trigger.get("value", 1) == 1:
            raise ValueError("Frame extrapolation should only be called when "\
                                    "trigger value is greater than 1")

        if self.extrapolated_frames % self.trigger["value"] == 0:
            raise ValueError("Frame extrapolation should only be called when inference was" \
                                                    "not run. check detection logic")

        xyxy_boxes = np.array([b["xyxy"] for b in boxes]).squeeze()
        diffs = np.diff(xyxy_boxes, axis=0)
        mean_diff = np.ma.average(diffs, axis=0, 
                            weights=range(len(diffs)))
        
        extrapolation_factor = self.extrapolated_frames / self.trigger["value"]
        next_xyxy_box = xyxy_boxes[-1] + (mean_diff * extrapolation_factor)
        
        return next_xyxy_box.tolist()

    @staticmethod
    def compile_onnx_model(model, imgsz):
        exported_model_path = model.export(format="openvino", simplify=True,
                                nms=True, imgsz=imgsz, batch=1, dynamic=True)
        ov_model = YOLOE(exported_model_path)
        return ov_model
    
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
        return []

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
