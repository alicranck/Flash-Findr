import yaml
import time
from typing import Sequence

from fastapi import APIRouter, Query, HTTPException, Body
from fastapi.responses import StreamingResponse
import cv2

from .engine import VideoInferenceEngine
from ..ml_core.tools.detection import OpenVocabularyDetector
from ..ml_core.tools.captioning import ImageCaptioningTool
from ..ml_core.tools.base_tool import BaseVisionTool
from ..ml_core.tools.pipeline import PipelineConfig, VisionPipeline
from ..utils.plotting import fast_plot


current_dir = __file__.rsplit("/", 1)[0]
with open(f"{current_dir}/../ml_core/configs/ov_detection.yaml", "r") as f:
    config = yaml.safe_load(f)


router = APIRouter()


# def stream_video_detection(detector: OpenVocabularyDetector, video_url: str):
#     """
#     Generator function that reads video frames, detects objects, and yields annotated JPEGs.
#     """

#     for results in detector.detect_stream(video_url):

#         t0 = time.time()        
#         if results.boxes:
#             annotated_frame = fast_plot(results.orig_img, results.boxes, 
#                                         results.names)
#         else:
#             annotated_frame = results.orig_img
#         t1 = time.time()
#         _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
#         t2 = time.time()
#         print(f"[INFO] Plotting time: {(t1 - t0)*1000:.2f} ms | Encoding time: {(t2 - t1)*1000:.2f} ms")
#         yield (b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        

def video_detection(pipeline: VisionPipeline, video_url: str,
                     vid_stride: int = 1):
    """
    Generator function that reads video frames, detects objects, and yields annotated JPEGs.
    """

    engine = VideoInferenceEngine(pipeline, video_url, frame_stride=vid_stride)
    return StreamingResponse(engine.run_inference(), media_type="multipart/x-mixed-replace; boundary=frame")


# @router.get("/stream/start")
# async def start_pipeline(
#     tool_types: str = Query(..., description="Comma-separated list of tool types (detection, captioning)"),
#     video_url: str = Query(..., description="URL of the video source (RTSP, MP4, etc.)"),
#     classes: str = Query(..., description="Comma-separated list of detection classes"),
#     output_type: str = Query("detection", description="Type of output: detection or captioning")
#     ):
#     """
#     Starts the real-time video processing stream using GET parameters.
#     """
#     class_list = [c.strip() for c in classes.split(',') if c.strip()]
    
#     try:
#         detector = OpenVocabularyDetector(config['model'])
#         detector.set_vocabulary(class_list)
#         return StreamingResponse(
#             video_detection(detector, video_url, config.get('vid_stride', 1)),
#             media_type="multipart/x-mixed-replace; boundary=frame"
#         )
#     except RuntimeError as e:
#         raise HTTPException(status_code=400, detail=str(e))
    

@router.post("/stream/start") # CHANGED FROM GET TO POST
async def start_pipeline(
    config: PipelineConfig = Body(..., description="JSON configuration for the vision pipeline")
):
    """
    Starts the real-time vision pipeline, validating configuration via a JSON request body, 
    and streams results.
    """
    requested_tools = config.tool_types
    pipeline_config = {
        'video_url': config.video_url,
        # Merge tool_settings into the main config dict
        **config.tool_settings 
    }
    
    try:
        pipeline = VisionPipeline(config=pipeline_config)
        
        # 3. Start the Stream
        return StreamingResponse(
            video_detection(pipeline, config.video_url, vid_stride=3),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
        
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Pipeline Configuration Error: {e}")