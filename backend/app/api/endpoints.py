import os
import uuid
from typing import Dict, Any
from pydantic import BaseModel, Field

from fastapi import APIRouter, Query, HTTPException, Body, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse

from .engine import VideoInferenceEngine
from ..ml_core.tools.pipeline import PipelineConfig, VisionPipeline
from .socket_manager import ConnectionManager
from ..utils.serialization import serialize_data

from ..core.session_manager import SessionManager

cache_dir = "./cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

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
    
    Args:
        session_config (SessionConfig): The configuration for the session, including video source and pipeline settings.
        
    Returns:
        JSONResponse: A JSON object containing the assigned `session_id`.
    """
    session_manager = SessionManager.get_instance()
    # Double check in case middleware didn't catch it (e.g. internal call)
    if session_manager.is_active():
         raise HTTPException(status_code=503, detail="System is busy with an active session.")

    try:
        session_id = str(uuid.uuid4())
        if not session_manager.start_session(session_id):
             raise HTTPException(status_code=503, detail="System is busy with an active session.")
             
        SESSION_STORE[session_id] = {
            "pipeline": None, 
            "video_url": session_config.video_url,
            "pipeline_configuration": session_config.pipeline_configuration
        }
        return JSONResponse(content={"session_id": session_id})
    except Exception as e:
        session_manager.end_session(session_id)
        raise HTTPException(status_code=400, detail=f"Session Initialization Error: {e}")


@router.post("/session/{session_id}/initialize_pipeline")
async def initialize_pipeline(session_id: str):
    """
    Triggers the loading of models for the specified session.
    This step is resource-intensive and must be called before streaming begins.
    
    Args:
        session_id (str): The ID of the session to initialize.
        
    Returns:
        JSONResponse: Status of the initialization.
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


@router.post("/upload_file")
async def upload_file(video: UploadFile = File(...)):
    file_path = os.path.join(cache_dir, video.filename)
    with open(file_path, "wb") as f:
        f.write(video.file.read())
    return JSONResponse(content={"file_path": file_path})
    

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

        async def stream_wrapper(generator):
            try:
                async for chunk in generator:
                    yield chunk
            finally:
                # Cleanup session when stream ends
                SessionManager.get_instance().end_session(session_id)
                if session_id in SESSION_STORE:
                    del SESSION_STORE[session_id]

        return StreamingResponse(
            stream_wrapper(engine.run_inference(on_data=on_data)),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
        
    except (RuntimeError, ValueError) as e:
        pipeline.unload_tools()
        SessionManager.get_instance().end_session(session_id)
        if session_id in SESSION_STORE:
             del SESSION_STORE[session_id]
        raise HTTPException(status_code=500, detail=f"Streaming Error: {e}")
