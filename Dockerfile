# syntax=docker/dockerfile:1
#
# Container image for Basalt: the engine, the Python server and its venv on an
# NVIDIA CUDA 13 runtime.
#
# Build from the repository root:
#   docker build -t basalt .
#
# The host needs an NVIDIA Blackwell GPU (compute capability 12.0) and a driver
# of the R580 branch or newer. The GPU reaches the container through the NVIDIA
# container runtime. Building needs no GPU: the engine is compiled for sm_120
# with no device present, and nothing runs CUDA during the build.
#
# The tree inside the image is a release archive's, with the venv already
# installed: bin/, serve/, tools/, data/ and .venv/ under one top directory. A
# binary finds its install root as the directory above itself, and reads the
# expert profiles from <install root>/data/profiles, execs the python of
# <install root>/.venv for `basalt serve` and looks for the basalt-vision
# sidecar beside itself. The top directory is the same fixed path in both stages
# because the venv's own scripts name it absolutely.
#
# Mounts:
#   /data    basalt_config.json (v2; tools/migrations/config_v1_to_v2.py converts
#            an older file) and, since HOME is pinned there, the server's folder
#            per model under ~/.cache/basalt: its engine log and, when the
#            conversation disk tier is on, that tier
#   /models  the .basalt model files

ARG CUDA_TAG=13.4.3

# ------------------------------------------------------------------ build stage
FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu24.04 AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates cmake python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/basalt
COPY . /opt/basalt/

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" \
    && python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------------------- final stage
FROM nvidia/cuda:${CUDA_TAG}-runtime-ubuntu24.04

# The base already carries the CUDA runtime and cuBLAS, the only CUDA libraries
# the engine links. python3 is what the venv's python points at.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/basalt

# The runtime tree: the binaries, the server and the tools it imports, the data/
# the engine reads, the venv, and the licences.
COPY --from=builder /opt/basalt/.venv            /opt/basalt/.venv
COPY --from=builder /opt/basalt/bin              /opt/basalt/bin
COPY --from=builder /opt/basalt/serve            /opt/basalt/serve
COPY --from=builder /opt/basalt/tools            /opt/basalt/tools
COPY --from=builder /opt/basalt/data             /opt/basalt/data
COPY --from=builder /opt/basalt/requirements.txt \
                  /opt/basalt/requirements.txt
COPY --from=builder /opt/basalt/LICENSE \
                  /opt/basalt/THIRD_PARTY_LICENSES \
                  /opt/basalt/

ENV HOME=/data \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/basalt/bin:$PATH

# Serve the config from the /data mount, bound to 0.0.0.0 so a mapped port
# reaches the server (loopback is the built-in default). Arguments given after
# the image name follow the entrypoint, e.g. --api-key KEY. The served port is
# server.port (8080) and the host mapping has to match it.
EXPOSE 8080

ENTRYPOINT ["basalt", "serve", "--config", "/data/basalt_config.json", "--host", "0.0.0.0"]
