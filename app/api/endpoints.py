import yaml
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
import cv2
from ..ml_core.detector import OpenVocabularyDetector

current_dir = __file__.rsplit("/", 1)[0]
with open(f"{current_dir}/../config.yaml", "r") as f:
    config = yaml.safe_load(f)


router = APIRouter()


def video_frame_generator(detector: OpenVocabularyDetector, video_url: str):
    """
    Generator function that reads video frames, detects objects, and yields annotated JPEGs.
    """

    for results in detector.detect_stream(video_url):
        
        if results.boxes:
            annotated_frame = results.plot() # Use YOLO's plot function for simplicity
        else:
            annotated_frame = results.orig_img

        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@router.get("/stream/start")
async def start_detection_stream(
    video_url: str = Query(..., description="URL of the video source (RTSP, MP4, etc.)"),
    classes: str = Query(..., description="Comma-separated list of detection classes")
):
    """
    Starts the real-time video processing stream using GET parameters.
    """
    class_list = [c.strip() for c in classes.split(',') if c.strip()]
    
    try:
        detector = OpenVocabularyDetector(config['ov_detection']['model'])
        detector.set_vocabulary(class_list)
        return StreamingResponse(
            video_frame_generator(detector, video_url),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))