from typing import List, Dict, Any

from pydantic import BaseModel, Field

from .detection import OpenVocabularyDetector
from .captioning import LlamaCppCaptioner
from ...utils.types import FrameContext


AVAILABLE_TOOL_TYPES = {
    'detection': OpenVocabularyDetector,
    'captioning': LlamaCppCaptioner
}

MODEL_IDS = {
    'detection': '/home/almog_elharar/almog/Flash-Findr/app/models/yoloe-11s-seg',
    'captioning': 'ggml-org/SmolVLM2-256M-Video-Instruct-GGUF:Q8_0'
}


class PipelineConfig(BaseModel):
    """Defines the expected structure for the JSON configuration body.
    In the future, this will be represented as layers of a DAG to allow for more complex pipelines."""    
    tool_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Nested dictionary of tool-specific static configuration " \
                        "(e.g., {'detection': {'vocabulary': ['person', 'car']}})."
    )


class VisionPipeline:
    """A modular vision processing pipeline that sequentially 
    applies a series of tools to input frames."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.tools = self._initialize_tools()

    def run_pipeline(self, frame: Any, context: FrameContext = None) -> Dict[str, Any]:

        data = {}
        for tool in self.tools:
            tool_results = tool.process(frame, data, context=context)
            data.update(tool_results)

        return frame, data

    def extrapolate_last(self, frame: Any) -> Dict[str, Any]:
        
        data = {}
        for tool in self.tools:
            tool_results = tool.extrapolate_last(frame)
            data.update(tool_results)

        return frame, data

    def _initialize_tools(self) -> List[Any]:

        tools = []
        for tool_type in self.config.tool_settings.keys():

            if tool_type not in AVAILABLE_TOOL_TYPES:
                raise ValueError(f"Tool type '{tool_type}' is not recognized.")
            
            tool_class = AVAILABLE_TOOL_TYPES[tool_type]
            tool_config = self.config.tool_settings.get(tool_type, {})
            tool_model_id = MODEL_IDS[tool_type]

            tool_instance = tool_class(model_id=tool_model_id, config=tool_config)
            tools.append(tool_instance)
        
        return tools

    def unload_tools(self):
        for tool in self.tools:
            tool.unload()

        
