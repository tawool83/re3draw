# Training image for the re3draw worker, for hosts that run containers (M3: one container per job).
# Modal builds the equivalent image from modal_app.py; this file is for any other GPU host.
#
#   cd packages/worker && docker build -f docker/Dockerfile -t re3draw-worker .
#   docker run --gpus all -v /data:/data re3draw-worker train /data/capture1 -o /data/capture1/splat
#   docker run --gpus all --entrypoint pytest re3draw-worker /opt/re3draw/tests -q   # self-test
#
# Why not the pytorch/pytorch image: gsplat's prebuilt wheels exist only for Python 3.10 (cp310),
# and those images ship Python 3.11, so gsplat would not install. Ubuntu 22.04 ships Python 3.10,
# and the torch 2.4 / CUDA 12.4 wheel bundles its own CUDA runtime, so a small CUDA base is enough.
# gsplat comes from its wheel index *in addition to* PyPI (--extra-index-url): with --index-url
# pip cannot find gsplat's own dependencies.
#
# Object segmentation (SAM 2) needs torch >= 2.5 and is not in this image; run `segment` in a
# separate environment first (see modal_app.py) - `train` picks up CAPTURE/masks when present.
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# git + cmake + build-essential + python3-dev (headers) are for building the .spz encoder.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python-is-python3 git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu's pip is too old for pyproject-only installs; upgrade it first.
RUN python -m pip install --upgrade pip     && pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124 \
    && pip install ninja \
    && pip install gsplat==1.5.3 --extra-index-url https://docs.gsplat.studio/whl/pt24cu124 \
    && pip install "git+https://github.com/nianticlabs/spz"

WORKDIR /opt/re3draw
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
RUN pip install ".[dev]"

ENTRYPOINT ["re3draw-worker"]
CMD ["--help"]
