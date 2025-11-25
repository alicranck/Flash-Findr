import asyncio
import cv2
import os
from ..ml_core.tools.pipeline import VisionPipeline
from ..utils.image_utils import color_histogram
from ..utils.types import FrameContext


class VideoInferenceEngine:

    def __init__(self, tool_pipeline: VisionPipeline, video_path: str):
        self.video_path = self._resolve_video_source(video_path)
        self.tool_pipeline = tool_pipeline
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

            scene_change_score = 0.0
            if self.last_frame is not None:
                 scene_change_score = self.hist_distance(self.last_frame, frame)
            else:
                 scene_change_score = 1.0

            context = FrameContext(frame_idx=self.last_frame_idx + 1,
                                   scene_change_score=scene_change_score,
                                   timestamp=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            processed_frame, data = self.tool_pipeline.run_pipeline(frame, context=context)
            
            if on_data:
                await on_data(data)
            
            self.last_frame_idx += 1
            self.last_frame = frame

            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
    @staticmethod
    def hist_distance(frame1, frame2) -> float:
        hist1 = color_histogram(frame1)
        hist2 = color_histogram(frame2)
        dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        return dist

    def _resolve_video_source(self, video_path: str) -> str:
        """
        Resolves the video source from a path or URL.
        Handles local files, YouTube links, and direct URLs.
        """
        if os.path.exists(video_path):
            return os.path.abspath(video_path)
        
        if "youtube.com" in video_path or "youtu.be" in video_path:
            raise ValueError("YouTube links are not supported.")

        return video_path


