# Multi-stage build for smaller final image
# Stage 1: Build llama.cpp
FROM python:3.11-slim AS llama-builder

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp \
    && cd llama.cpp \
    && cmake -B build -DGGML_CUDA=OFF \
    && cmake --build build --config Release -j $(nproc)

# Stage 2: Final runtime image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LLAMA_SERVER_PATH=/usr/local/bin/llama-server

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    libcurl4 \
    && rm -rf /var/lib/apt/lists/*

# Copy llama-server binary from builder stage
COPY --from=llama-builder /tmp/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Set PYTHONPATH to include /app
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8008

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8008"]