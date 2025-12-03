from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .api import endpoints as api_endpoints
from .api.session_manager import SessionManager

app = FastAPI(
    title="Flash-Findr: Real-Time OV Detector",
    description="Zero-shot object detection microservice using YOLO-Everything.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the endpoints that handle the stream and detection logic
app.include_router(api_endpoints.router, tags=["Detection Stream"])

@app.middleware("http")
async def session_guard(request: Request, call_next):
    """
    Middleware to reject new session initialization if a session is already active.
    """
    if request.url.path == "/session/init" and request.method == "POST":
        manager = SessionManager.get_instance()
        if manager.is_active():
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "System is busy with an active session. Please try again later."}
            )
            
    response = await call_next(request)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8008, reload=True)