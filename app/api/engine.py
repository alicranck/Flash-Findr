
import asyncio
import cv2
from pydantic import BaseModel, Field
from ..ml_core.tools.pipeline import VisionPipeline, PipelineConfig
from ..utils.plotting import fast_plot
from ..utils.image_utils import color_histogram


class SessionConfig(BaseModel):
    """Defines the configuration for a streaming session."""
    video_url: str = Field(..., description="URL or path to the video source")
    video_stride: int = Field(-1, description="Frame stride for inference")
    frame_distance: int = Field(-1, description="Distance threshold for frame processing")
    pipeline: PipelineConfig = Field(..., description="Pipeline configuration")


class VideoInferenceEngine:

    def __init__(self, tool_pipeline: VisionPipeline, video_path: str,
                  frame_stride: int = -1, dist_th: float = -1):
        self.video_path = video_path
        self.tool_pipeline = tool_pipeline

        assert (frame_stride >= 1) ^ (dist_th > 0), \
                    "Exactly one of frame_stride or dist_th must be set."
        self.frame_stride = frame_stride
        self.dist_th = dist_th

        self.last_frame_idx = -1
        self.last_frame = None

    async def run_inference(self, on_data=None):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Error opening video stream or file: {self.video_path}")

        while True:
            # Yield control to event loop to allow WS connections
            await asyncio.sleep(0)
            
            ret, frame = cap.read()
            if not ret:
                break

            if self.should_run(frame):
                processed_frame, data = self.tool_pipeline.run_pipeline(frame)
            else:
                processed_frame, data = self.tool_pipeline.extrapolate_last(frame)
            
            # Broadcast data if callback is provided
            if on_data:
                await on_data(data)

            # For frontend visualization, we stream the CLEAN frame (no fast_plot)
            # The frontend will draw the boxes based on the data sent via WS.
            annotated_frame = processed_frame 
            
            self.last_frame_idx += 1
            self.last_frame = frame

            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
    def should_run(self, frame) -> bool:
        
        frame_idx = self.last_frame_idx + 1

        if frame_idx == 0:
            return True
        if self.frame_stride > 0:
            return (frame_idx % self.frame_stride) == 0
        else:
            dist = self.hist_distance(self.last_frame, frame)
            return dist >= self.dist_th
            
    @staticmethod
    def hist_distance(frame1, frame2) -> float:
        hist1 = color_histogram(frame1)
        hist2 = color_histogram(frame2)
        dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        return dist


