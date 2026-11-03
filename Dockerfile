# syntax=docker/dockerfile:1
#
# Container image for Basalt: the engine, the Python server and its venv under one
# fixed source tree, on an NVIDIA CUDA 13 runtime.
#
# Build from the repository root:
#   docker build -t basalt .
#
# The host needs an NVIDIA Blackwell GPU (compute capability 12.0) and a driver new
# enough for CUDA 13. The GPU reaches the container through the NVIDIA container
# runtime, or by mapping /dev/nvidia0, /dev/nvidiactl, /dev/nvidia-uvm and
# /dev/nvidia-uvm-tools into it.
#
# Mounts:
#   /data    basalt_config.json, and the server's cache root ($HOME/.cache/basalt,
#            pinned here by HOME): the tokenizer and vision files it unpacks from
#            the model pack, the conversation disk tier and the engine log
#   /models  the .basalt model files

ARG CUDA_TAG=13.4.3

# ------------------------------------------------------------------ build stage
FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu24.04 AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates cmake python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# The engine bakes the cmake source directory into the binary at compile time
# (BASALT_SOURCE_DIR): at runtime it reads data/profiles/ from there and `basalt
# serve` chdirs into it and execs .venv/bin/python there. So the source tree must
# live at a fixed path, and that same path is reproduced in the final image.
WORKDIR /opt/basalt
COPY . /opt/basalt/

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" \
    && python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------- final stage
FROM nvidia/cuda:${CUDA_TAG}-runtime-ubuntu24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates python3 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/basalt

# Runtime tree: binaries (basalt + basalt-vision sidecar must sit next to each
# other), the Python server, the tools the server imports, the data/ expert
# profiles, and the venv the server runs under.
COPY --from=builder /opt/basalt/.venv    /opt/basalt/.venv
COPY --from=builder /opt/basalt/bin      /opt/basalt/bin
COPY --from=builder /opt/basalt/serve    /opt/basalt/serve
COPY --from=builder /opt/basalt/tools    /opt/basalt/tools
COPY --from=builder /opt/basalt/data     /opt/basalt/data
COPY --from=builder /opt/basalt/requirements.txt     /opt/basalt/requirements.txt

ENV HOME=/data \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/basalt/bin:$PATH

# Serve with the config from the /data mount; bind 0.0.0.0 by default so the
# port mapping reaches the server (the built-in default is loopback) — pass
# --host on the command line to override. If the API is exposed on an
# untrusted network, add --api-key KEY (it cannot live in the config file).
EXPOSE 8080

ENTRYPOINT ["basalt", "serve", "--config", "/data/basalt_config.json", "--host", "0.0.0.0"]
