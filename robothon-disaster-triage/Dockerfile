# Disaster Triage Robothon — Dockerfile
# Reproducible build with headless rendering support via Xvfb.
# Build:  docker build -t disaster-triage-robothon .
# Run:    docker run --rm disaster-triage-robothon python -m src.cli inspect

FROM python:3.11-slim AS base

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MUJOCO_GL=egl \
    DISPLAY=:99

# System dependencies:
#   xvfb         — virtual framebuffer for headless OpenGL rendering
#   libgl1, libgllib1-mesa-glx, libegl1-mesa, libgles2-mesa
#                — OpenGL libraries (mujoco.Renderer needs them)
#   ffmpeg       — video encoding (imageio-ffmpeg bundles one, but system is more reliable)
#   libosmesa6   — fallback OSMesa backend
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb \
        libgl1 \
        libgl1-mesa-glx \
        libegl1-mesa \
        libgles2-mesa \
        libosmesa6 \
        ffmpeg \
        xauth \
        x11-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Pre-create runtime directories (so non-root user can write to them)
RUN mkdir -p checkpoints demos logs && \
    chmod -R 777 checkpoints demos logs

# Smoke-test on container start: render one frame to verify the renderer
# works, then drop into a bash shell for interactive use.
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["xvfb-run -a python -m src.cli inspect --steps 50 && echo '\\n[Container] Smoketest passed.\\n' && exec bash"]

# Default command can be overridden:
#   docker run --rm disaster-triage-robothon python -m src.cli scripted --max-steps 500
#   docker run --rm disaster-triage-robothon python -m src.cli rl-train --smoketest
#   docker run --rm -v $(pwd)/demos:/app/demos disaster-triage-robothon xvfb-run -a python -m src.cli demo --duration 60
