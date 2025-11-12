import os
from ultralytics import YOLOE # type: ignore
from PIL.Image import Image as PILImage

import openvino.properties.hint as hints
from openvino import Core

current_dir = __file__.rsplit("/", 1)[0]

class OpenVocabularyDetector:

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = self._init_model(model_id)
        self.openvino_model = None
        self.vocab = []

    def set_vocabulary(self, vocab_list: list[str]):
        self.vocab = vocab_list
        pos_embeddings = self.model.get_text_pe(vocab_list)
        self.model.set_classes(self.vocab, pos_embeddings)
        self.openvino_model = self.compile_openvino_model()

    def detect(self, image: PILImage, confidence_threshold: float = 0.25):
        # Move to ONNX for faster inference
        assert self.openvino_model is not None, \
                "ONNX model is not initialized. Call set_vocabulary() first."
        results = self.openvino_model.predict(image)
        return results
    
    def detect_stream(self, video_url: str, confidence_threshold: float = 0.25):
                # Move to ONNX for faster inference
        assert self.openvino_model is not None, \
                "ONNX model is not initialized. Call set_vocabulary() first."
        results = self.openvino_model.predict(source=video_url, stream=True)
        return results

    def _init_model(self, model_id: str):
        # check if file exists else download to models/
        if not os.path.isfile(model_id):
            model_path = os.path.join(f"{current_dir}/../../models", model_id)
            model = YOLOE(model_path)
        else:
            model = YOLOE(model_id)

        return model

    def compile_openvino_model(self):
        core = Core()
        config = {"PERFORMANCE_HINT": "THROUGHPUT",}

        self.model.export(format="openvino", half=True, dynamic=True, batch=16)
        export_xml_path = os.path.join(current_dir, "../../models/",
                                        self.model_id.split(".")[0] + "_openvino_model",
                                          self.model_id.split(".")[0] + ".xml")
        openvino_model = core.compile_model(export_xml_path, "CPU", config)
        
        return openvino_model
