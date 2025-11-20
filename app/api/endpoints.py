import yaml
import uuid
from typing import Dict, Any
from fastapi import APIRouter, Query, HTTPException, Body, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
import cv2

from .engine import VideoInferenceEngine, SessionConfig
from ..ml_core.tools.pipeline import PipelineConfig, VisionPipeline
from ..utils.plotting import fast_plot
from .socket_manager import ConnectionManager
from ..utils.serialization import serialize_data


router = APIRouter()

# In-memory session store: {session_id: SessionConfig}
SESSION_STORE: Dict[str, SessionConfig] = {}

# WebSocket Manager
manager = ConnectionManager()

@router.post("/session/init")
async def init_session(
    config: SessionConfig = Body(..., description="JSON configuration for the vision session")
):
    """
    Initializes a streaming session with the provided configuration.
    Returns a session_id that can be used to consume the stream via GET.
    """
    try:
        session_id = str(uuid.uuid4())
        SESSION_STORE[session_id] = config
        return JSONResponse(content={"session_id": session_id})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Session Initialization Error: {e}")


@router.websocket("/ws/stream/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)
    try:
        while True:
            # Keep connection alive, maybe listen for client events later
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(session_id, websocket)


@router.get("/stream/{session_id}")
async def stream_video(session_id: str):
    """
    Streams the video for a valid session_id.
    """
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    
    session_config = SESSION_STORE[session_id]
    
    try:
        pipeline = VisionPipeline(config=session_config.pipeline)
        engine = VideoInferenceEngine(
            tool_pipeline=pipeline,
            video_path=session_config.video_url,
            frame_stride=session_config.video_stride,
            dist_th=session_config.frame_distance
        )
        
        async def on_data(data):
            # Serialize and broadcast to WebSocket
            json_data = serialize_data(data)
            await manager.broadcast(session_id, json_data)

        return StreamingResponse(
            engine.run_inference(on_data=on_data),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
        
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Streaming Error: {e}")
        
        return StreamingResponse(
            engine.run_inference(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
        
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Pipeline Configuration Error: {e}")