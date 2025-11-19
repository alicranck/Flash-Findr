from abc import ABC, abstractmethod
import numpy as np
import torch

from ...utils.types import ImageHandle, List, Any
from ...utils.image_utils import load_image_opencv


class ToolKey:
    """
    A descriptor for a data key provided or required by a tool.
    This is the core of the "manifest."
    """
    def __init__(self, key_name: str, data_type: Any, description: str,
                 required: bool = False):
        self.key_name = key_name
        self.data_type = data_type
        self.description = description
        self.required = required

    def __repr__(self):
        return f"ToolKey(key='{self.key_name}', type={self.data_type.__name__}, required={self.required})"


class BaseVisionTool(ABC):
    """
    Abstract Base Class for a modular vision tool.
    
    This class handles the common workflow:
    1. State (loaded/unloaded)
    2. Device management
    3. The load() -> warmup() orchestration
    4. The unload() orchestration
    5. The preprocess() -> inference() -> postprocess() pipeline
    """
    def __init__(self, model_id: str, config: dict, 
                 device: str = 'cpu'):
        """Initializes the tool with configuration, but does not load the model."""
        self.model_id : str = model_id
        self.device: str = device

        self.model : Any
        self.loaded: bool = False
        self.tool_name: str = self.__class__.__name__

        self.load_tool(config)  # Optionally load the model during initialization

    def load_tool(self, config):
        """
        Public method to load, verify, and warmup the model.
        This is the common "init model" flow.
        """
        if self.loaded:
            print(f"INFO: {self.tool_name} is already loaded.")
            return

        print(f"INFO: Loading {self.tool_name} with model: {self.model_id}...")

        for key in self.config_keys:
            if key.required:
                if key.key_name not in config:
                    raise ValueError(f"ERROR: Missing required config key '{key.key_name}' for {self.tool_name}.")
                # assert isinstance(config[key.key_name], key.data_type), \
                #     f"ERROR: Config key '{key.key_name}' must be of type {key.data_type.__name__}."
                
        self._configure(config)

        try:
            self.model = self._load_model()
            self._warmup()
            self.loaded = True
            print(f"INFO: {self.tool_name} successfully loaded and warmed up on {self.device}.")
            
        except Exception as e:
            self.model = None
            self.loaded = False
            print(f"ERROR: Failed to load {self.tool_name}. Error: {e}")
            raise

    def unload_tool(self):
        """
        Public method to clear the model from device memory.
        This is the common "teardown" flow.
        """
        if self.model:
            del self.model
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        self.model = None
        self.loaded = False
        print(f"INFO: {self.tool_name} unloaded and cleared from {self.device}.")

    def process(self, frame_handle: ImageHandle) -> dict:
        """
        Public method to run the full inference pipeline.
        This is the common "process frame" flow.
        """
        if not self.loaded:
            raise RuntimeError(f"ERROR: {self.tool_name} is not loaded. Call .load_tool() first.")

        frame = load_image_opencv(frame_handle)
        model_input = self.preprocess(frame)
        
        with torch.no_grad():
            raw_output = self.inference(model_input)

        updated_data = self.postprocess(raw_output, frame.shape)

        return updated_data

    def _configure(self, config: dict):
        """Child implements tool-specific configuration logic."""
        pass

    @abstractmethod
    def _load_model(self) -> Any:
        """Child implements the specific model loading logic (e.g., YOLO(path))."""
        pass

    @abstractmethod
    def _warmup(self):
        """Child implements a dummy inference run to warm up the model."""
        pass

    @abstractmethod
    def preprocess(self, frame: np.ndarray) -> Any:
        """Child implements frame-to-tensor logic (resize, normalize, to-device)."""
        pass

    @abstractmethod
    def inference(self, model_inputs: Any) -> Any:
        """Child implements the raw model.forward() call."""
        pass

    @abstractmethod
    def postprocess(self, raw_output: Any, original_shape: tuple) -> dict:
        """Child implements logic to parse raw_output."""
        pass

    @property
    @abstractmethod
    def output_keys(self) -> List[ToolKey]:
        """
        Class property declaring the data keys this tool *produces* and adds to the 'data' dictionary.
        """
        pass

    @property
    @abstractmethod
    def processing_input_keys(self) -> List[ToolKey]:
        """
        Class property declaring the data keys this tool *requires* to be present in the 'data' dictionary to run.
        """
        pass

    @property
    @abstractmethod
    def config_keys(self) -> List[ToolKey]:
        """
        Class property declaring the configuration keys this tool *requires* to be present in the 'config' dictionary to run.
        """
        pass