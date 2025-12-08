# Base image for Jetson Nano (JetPack 4.6) containing PyTorch & CUDA
FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

# Install system dependencies (OpenCV requires these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip3 install --upgrade pip

# Install Python dependencies
# Note: PyTorch & Torchvision are already in the base image, so we comment them out if present in requirements,
# or we trust pip to skip them if versions match. 
# Here we copy requirements.txt and install.
COPY requirements.txt .

# We filter out torch/torchvision from requirements.txt to avoid pip trying to download x86 wheels or compiling from source
RUN grep -v "torch" requirements.txt > requirements_jetson.txt && \
    pip3 install -r requirements_jetson.txt

# Copy source code
COPY . .

# Default command (can be overridden)
CMD ["python3", "inference.py", "--camera_id", "0", "--mqtt_broker", "broker.hivemq.com"]
