# 🐳 Docker Deployment Guide for Jetson Nano

This guide explains how to deploy the Cassava OOD Detection system on an NVIDIA Jetson Nano using Docker.

## Prerequisites
1.  **Jetson Nano** with JetPack 4.6 (L4T 32.7.1).
2.  **Docker** & **NVIDIA Container Runtime** (Usually pre-installed on JetPack).
3.  **USB Camera** connected (checked via `ls /dev/video0`).

> [!IMPORTANT]
> **Architecture Warning:** This Dockerfile uses a base image (`l4t-pytorch`) designed specifically for **Jetson (ARM64)**.
> You **CANNOT** build this image on a standard Windows/Linux PC (x86_64) and transfer it to the Nano. It will fail with an "Exec format error".
> **You MUST build this image directly on the Jetson Nano.**

## Setup

1.  **Clone the repository on Jetson Nano:**
    ```bash
    git clone https://github.com/thanquannguyen/cassava_OOD.git
    cd cassava_OOD
    ```

    > [!TIP]
    > **If your Git LFS quota is exceeded:**
    > 1.  Clone without LFS files:
    >     ```bash
    >     GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/thanquannguyen/cassava_OOD.git
    >     cd cassava_OOD
    >     ```
    > 2.  **Use WinSCP (or SCP)** on your computer to copy the large files (e.g., `checkpoints/*.pth`, `data/`) from your local machine to the `cassava_OOD` folder on the Jetson Nano.
    > 3.  WinSCP will ask to overwrite the small "pointer" files. Select **Yes**.
    ```

2.  **Build the Docker Image:**
    ```bash
    sudo docker build -t cassava-ood .
    ```
    *Note: This might take a while as it downloads the base image (~1GB).*

## Running

### Option 1: Using Docker Compose (Recommended)
Make sure `docker-compose` is installed. If not: `sudo apt-get install docker-compose`.

Run the system:
```bash
sudo docker-compose up
```

### Option 2: Using Docker Command Line
```bash
sudo docker run -it --rm \
    --runtime nvidia \
    --network host \
    --network host \
    --device /dev/video0 \
    -v $(pwd):/app \
    -v $(pwd)/checkpoints:/app/checkpoints \
    cassava-ood \
    python3 inference.py --camera_id 0 --mqtt_broker broker.hivemq.com
```

### Option 3: Simulation Mode (No Camera)
If you don't have a camera, loop a test image:
```bash
sudo docker run -it --rm \
    --runtime nvidia \
    --network host \
    -v $(pwd):/app \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -v $(pwd)/torch_cache:/root/.cache/torch \
    cassava-ood \
    python3 inference.py --image_path demo_id.jpg --mqtt_broker broker.hivemq.com --loop --headless
```

> [!TIP]
> **Development Mode:** thanks to `-v $(pwd):/app`, any change you make to the code (e.g., editing `inference.py`) will take effect immediately. **You do NOT need to rebuild** unless you add new libraries to `requirements.txt`.

## Troubleshooting
-   **Camera not found:** Check if `/dev/video0` exists. Try `ls -l /dev/video*`.
-   **Permission denied:** Run docker with `sudo` or add your user to the `docker` group.
-   **Torch error:** Ensure you are using the correct base image in `Dockerfile` matching your JetPack version. Check `jtop` or `cat /etc/nv_tegra_release`.
-   **Error "Unsupported config option: runtime":**
    -   This means your `docker-compose` version is too old (installed via `apt`).
    -   **Fix:** Use **Option 2** (docker run) above, OR upgrade docker-compose:
        ```bash
        sudo apt-get remove docker-compose
        sudo pip3 install docker-compose
        ```
-   **Error "pull access denied for cassava-ood":**
    -   This means you haven't built the image yet, so Docker is trying (and failing) to find it online.
    -   **Fix:** Run the build command first:
        ```bash
        sudo docker build -t cassava-ood .
        ```
