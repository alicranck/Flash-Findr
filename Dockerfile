# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LLAMA_SERVER_PATH=/usr/local/bin/llama-server

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp with CUDA support
WORKDIR /tmp
RUN git clone https://github.com/ggerganov/llama.cpp.git \
    && cd llama.cpp \
    && make LLAMA_CUDA=1 \
    && cp llama-server /usr/local/bin/ \
    && cd .. \
    && rm -rf llama.cpp

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8008

# Default command (can be overridden by docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8008"]
