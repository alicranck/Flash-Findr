from typing import List, Dict, Any

from pydantic import BaseModel, Field

from .detection import OpenVocabularyDetector
from .captioning import ImageCaptioningTool

AVAILABLE_TOOL_TYPES = {
    'detection': OpenVocabularyDetector,
    'captioning': ImageCaptioningTool
}


class PipelineConfig(BaseModel):
    """Defines the expected structure for the JSON configuration body."""
    
    video_url: str = Field(..., description="URL of the video source (RTSP, MP4, etc.)")
    tool_types: List[str] = Field(..., description="List of tool identifiers to activate (e.g., ['detection', 'segmentation']).")
    
    # Tool settings are now nested under the tool's name.
    tool_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Nested dictionary of tool-specific static configuration (e.g., {'detection': {'vocabulary': ['person', 'car']}})."
    )


class VisionPipeline:

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.tools = self._initialize_tools()

    def run_pipeline(self, frame: Any) -> Dict[str, Any]:

        processed_frame = frame

        for tool in self.tools:
            inputs = tool.preprocess(processed_frame)
            outputs = tool.inference(inputs)
            processed_frame = tool.postprocess(outputs, processed_frame)

        return processed_frame

    def _initialize_tools(self) -> List[Any]:

        self._verify_config()

        tools = []
        for tool_type in self.config.tool_types:
            if tool_type not in AVAILABLE_TOOL_TYPES:
                raise ValueError(f"Tool type '{tool_type}' is not recognized.")
            
            tool_class = AVAILABLE_TOOL_TYPES[tool_type]
            tool_config = self.config.tool_settings.get(tool_type, {})
            tool_instance = tool_class(config=tool_config)
            tools.append(tool_instance)
        
        return tools
    
    def _verify_config(self):
        """Verifies that all required config keys for each tool are present."""
        for tool in self.tools:
            for key in tool.config_keys():
                if key.required and key.key_name not in self.config.get(tool.tool_name, {}):
                    raise ValueError(f"Missing required config key '{key.key_name}' for tool '{tool.tool_name}'.")
        
