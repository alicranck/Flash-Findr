from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from .api import endpoints as api_endpoints

current_dir = __file__.rsplit("/", 1)[0]

templates = Jinja2Templates(directory=f"{current_dir}/../frontend/templates")

app = FastAPI(
    title="Flash-Findr: Real-Time OV Detector",
    description="Zero-shot object detection microservice using YOLO-Everything.",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory=f"{current_dir}/../frontend/static"), name="static")

@app.get("/", response_class=HTMLResponse, summary="Serve Web UI")
async def serve_ui(request: Request):
    """
    Serves the main application interface (index.html).
    """
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "title": "Flash-Findr Live Detection"}
    )

# Include the endpoints that handle the stream and detection logic
app.include_router(api_endpoints.router, tags=["Detection Stream"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8008, reload=True)