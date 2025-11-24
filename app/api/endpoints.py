import uuid
from typing import Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from fastapi import APIRouter, Query, HTTPException, Body, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse

from .engine import VideoInferenceEngine
from ..ml_core.tools.pipeline import PipelineConfig, VisionPipeline
from .socket_manager import ConnectionManager
from ..utils.serialization import serialize_data


router = APIRouter()


class SessionConfig(BaseModel):
    """Defines a streaming session."""
    video_url: str = Field(..., description="URL or path to the video source")
    pipeline_configuration: PipelineConfig = Field(..., description="Pipeline configuration")


# In-memory session store: {session_id: Session}
SESSION_STORE: Dict[str, Dict[str, Any]] = {}

# WebSocket Manager
manager = ConnectionManager()

@router.post("/session/init")
async def init_session(
    session_config: SessionConfig = Body(..., description="JSON configuration for the vision session")
):
    """
    Initializes a streaming session with the provided configuration.
    Returns a session_id that can be used to consume the stream via GET.
    """
    try:
        session_id = str(uuid.uuid4())
        # Store config for later initialization
        SESSION_STORE[session_id] = {
            "pipeline": None, 
            "video_url": session_config.video_url,
            "pipeline_configuration": session_config.pipeline_configuration
        }
        return JSONResponse(content={"session_id": session_id})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Session Initialization Error: {e}")


@router.post("/session/{session_id}/initialize_pipeline")
async def initialize_pipeline(session_id: str):
    """
    Triggers the loading of models for the specified session.
    This step must be called before streaming.
    """
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    session = SESSION_STORE[session_id]
    
    if session["pipeline"] is not None:
        return JSONResponse(content={"status": "already_initialized", "session_id": session_id})

    try:
        session["pipeline"] = VisionPipeline(config=session["pipeline_configuration"])
        return JSONResponse(content={"status": "initialized", "session_id": session_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline Initialization Failed: {e}")


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
    Requires pipeline to be initialized first.
    """
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    
    session = SESSION_STORE[session_id]
    pipeline = session["pipeline"]
    video_url = session["video_url"]

    if pipeline is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized. Call /initialize_pipeline first.")
    
    try:
        engine = VideoInferenceEngine(
            tool_pipeline=pipeline,
            video_path=video_url,
        )
        
        async def on_data(data):
            json_data = serialize_data(data)
            await manager.broadcast(session_id, json_data)

        return StreamingResponse(
            engine.run_inference(on_data=on_data),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
        
    except (RuntimeError, ValueError) as e:
        pipeline.unload_tools()
        raise HTTPException(status_code=500, detail=f"Streaming Error: {e}")
