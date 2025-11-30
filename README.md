# Flash-Findr: Real-Time Open-Vocabulary Object Detection

Flash-Findr is a high-performance, real-time object detection microservice that leverages YOLO-Everything for zero-shot detection with custom vocabularies. It features a modern web interface with smooth video streaming, and image captioning.

## Features

- **Open-Vocabulary Detection**: Detect any object by simply providing a list of class names
- **Real-Time Streaming**: Designed to run in real time on CPU
- **Image Captioning**: Optional scene captioning using vision-language models
- **GPU Acceleration**: automatic CPU / CUDA runtime support

## Architecture

```
Flash-Findr/
├── app/
│   ├── api/              # FastAPI endpoints and streaming engine
│   ├── ml_core/          # Computer Vision tools
│   ├── utils/            # Helper utilities
│   └── main.py           # Application entry point
├── frontend/             # Web UI (HTML, CSS, JavaScript)
```

## Setup Instructions

### Prerequisites

- Python 3.10+
- (Optional) NVIDIA GPU with CUDA support for acceleration

### Option 1: Local Setup (Without Docker)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Flash-Findr
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python -m app.main
   ```

5. **Access the web interface**
   
   Open your browser and navigate to: `http://localhost:8008`

### Option 2: Docker Setup

#### CPU-Only Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose --profile cpu up --build
   ```

2. **Access the application**
   
   Navigate to: `http://localhost:8008`

#### GPU-Accelerated Deployment

1. **Ensure NVIDIA Docker runtime is installed**
   ```bash
   # Install nvidia-docker2
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Build and run with GPU support**
   ```bash
   docker-compose --profile gpu up --build
   ```

3. **Access the application**
   
   Navigate to: `http://localhost:8008`

## Development

### Project Structure

- **`app/api/engine.py`**: Video processing and streaming engine
- **`app/ml_core/tools/`**: Modular vision tools (detection, captioning)
- **`app/utils/tracking.py`**: Kalman Filter implementation
- **`frontend/static/main.js`**: Client-side UI logic with Konva.js

### Adding New Tools

1. Create a new tool class inheriting from `BaseVisionTool`
2. Implement required methods: `_load_model`, `inference`, `postprocess`
3. Register in `AVAILABLE_TOOL_TYPES` in `pipeline.py`
4. Add configuration YAML in `app/ml_core/configs/`

## Troubleshooting

**Models not loading**: Ensure the `models/` directory exists and has write permissions

**GPU not detected**: Verify CUDA installation with `nvidia-smi` and check Docker GPU runtime

**Slow streaming**: Reduce detection stride or lower image resolution in tool settings

**WebSocket disconnects**: Check firewall settings and ensure port 8008 is accessible

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection
- [SORT](https://github.com/abewley/sort) for object tracking
- [Konva.js](https://konvajs.org/) for canvas rendering
