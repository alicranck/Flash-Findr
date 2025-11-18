
from typing import Sequence
import yaml
import time
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
import cv2
from ..ml_core.tools.base_tool import BaseVisionTool
from ..utils.plotting import fast_plot


class VideoInferenceEngine:

    def __init__(self, tool_pipeline: Sequence[BaseVisionTool], video_path: str,
                  frame_stride: int = 3):
        self.video_path = video_path
        self.tool_pipeline = tool_pipeline
        self.frame_stride = frame_stride

    def run_inference(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Error opening video stream or file: {self.video_path}")

        frame_count = 0
        while True:
            
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.frame_stride == 0:

                processed_frame = self.tool_pipeline.run_pipeline(frame)
                annotated_frame = processed_frame
                frame_count += 1        
            else:
                frame_count += 1
                continue

            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


