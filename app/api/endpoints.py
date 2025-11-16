import yaml
import time
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
import cv2
from ..ml_core.tools.detection import OpenVocabularyDetector
from ..utils.plotting import fast_plot

current_dir = __file__.rsplit("/", 1)[0]
with open(f"{current_dir}/../ml_core/configs/ov_detection.yaml", "r") as f:
    config = yaml.safe_load(f)


router = APIRouter()


def stream_video_detection(detector: OpenVocabularyDetector, video_url: str):
    """
    Generator function that reads video frames, detects objects, and yields annotated JPEGs.
    """

    for results in detector.detect_stream(video_url):

        t0 = time.time()        
        if results.boxes:
            annotated_frame = fast_plot(results.orig_img, results.boxes, 
                                        results.names)
        else:
            annotated_frame = results.orig_img
        t1 = time.time()
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        t2 = time.time()
        print(f"[INFO] Plotting time: {(t1 - t0)*1000:.2f} ms | Encoding time: {(t2 - t1)*1000:.2f} ms")
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        

def video_detection(detector: OpenVocabularyDetector, video_url: str, vid_stride: int = 1):
    """
    Generator function that reads video frames, detects objects, and yields annotated JPEGs.
    """

    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video stream or file: {video_url}")

    frame_count = 0
    while True:
        
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % vid_stride == 0:
            results = detector.detect(frame, confidence_threshold=config['confidence_threshold'])[0]
            annotated_frame = fast_plot(results.orig_img, results.boxes, results.names)
            frame_count += 1        
        else:
            frame_count += 1
            continue

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
        detector = OpenVocabularyDetector(config['model'])
        detector.set_vocabulary(class_list)
        return StreamingResponse(
            video_detection(detector, video_url, config.get('vid_stride', 1)),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))