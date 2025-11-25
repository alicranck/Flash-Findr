import os
import sys
import yaml
from ultralytics import YOLOE
from transformers import AutoProcessor, AutoTokenizer
from optimum.intel.openvino.modeling_visual_language import OVModelForVisualCausalLM

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(BASE_DIR, "app/ml_core/configs")
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")


def load_config(filename):
    path = os.path.join(CONFIGS_DIR, filename)
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def prepare_yolo():
    config = load_config("ov_detection.yaml")
    model_id = config.get("model", "yoloe-11s-seg.pt")
    
    print(f"Preparing YOLO model: {model_id}...")
    model_path = os.path.join(MODELS_DIR, model_id)
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    # This will download the model if not present
    model = YOLOE(model_id)
    model.save(model_path)
    print("YOLO model ready.")

def prepare_captioning():
    config = load_config("captioning.yaml")
    model_id = config.get("model", "HuggingFaceTB/SmolVLM2-256M-Instruct-OV")
    
    print(f"Preparing Captioning model: {model_id}...")
    try:
        OVModelForVisualCausalLM.from_pretrained(model_id)
        AutoProcessor.from_pretrained(model_id)
        AutoTokenizer.from_pretrained(model_id)
        print("Captioning model ready.")
    except Exception as e:
        print(f"Error preparing captioning model: {e}")
        pass

if __name__ == "__main__":
    print("Starting model preparation...")
    try:
        prepare_yolo()
        prepare_captioning()
        print("Model preparation complete.")
    except Exception as e:
        print(f"Model preparation failed: {e}")
        sys.exit(1)
