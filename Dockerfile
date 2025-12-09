# Base image for Jetson Nano (JetPack 4.6) containing PyTorch & CUDA
FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

# Install system dependencies (OpenCV requires these)
# Fix: Remove broken kitware repo from sources.list (it's not in sources.list.d)
RUN sed -i '/kitware/d' /etc/apt/sources.list && \
    rm -rf /etc/apt/sources.list.d/kitware* && \
    apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    python3-opencv \
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

# We filter out torch/torchvision AND opencv-python from requirements.txt 
# Reason: torch is in base image; opencv-python fails to build on ARM, so we use python3-opencv from apt above
RUN grep -v "torch" requirements.txt | grep -v "opencv-python" > requirements_jetson.txt && \
    pip3 install -r requirements_jetson.txt

# Copy source code
COPY . .

# Default command (can be overridden)
CMD ["python3", "inference.py", "--camera_id", "0", "--mqtt_broker", "broker.hivemq.com"]
