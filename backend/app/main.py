# import uvicorn
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .api import endpoints as api_endpoints
from .api import indexer as indexer_endpoints
from .api.session_manager import SessionManager


app = FastAPI(
    title="Flash-Findr: Real-Time OV Detector",
    description="Zero-shot object detection microservice using YOLO-Everything.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://base-frontend-image-119510690946.europe-west3.run.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the endpoints that handle the stream and detection logic
app.include_router(api_endpoints.router, tags=["Detection Stream"])
app.include_router(indexer_endpoints.router, prefix="/indexer", tags=["Video Indexer"])