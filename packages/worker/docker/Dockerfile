# Training image for the re3draw worker. Not needed to *develop* M2 - on a rented GPU box it is
# faster to take that host's PyTorch template and run scripts/setup-gpu.sh. This image exists for
# M3, where the API launches one container per job and start-up time is the user's waiting time.
#
#   cd packages/worker && docker build -f docker/Dockerfile -t re3draw-worker .
#   docker run --gpus all -v /data:/data re3draw-worker train /data/capture1 -o /data/capture1/splat
#   docker run --gpus all --entrypoint pytest re3draw-worker /opt/re3draw/tests -q   # self-test
#
# CUDA 12.4 / PyTorch 2.4 is chosen because gsplat publishes prebuilt wheels for that pair, which
# keeps the build to seconds instead of a source compile of the CUDA kernels.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# git + cmake are for the .spz encoder; libgl/libglib are OpenCV's runtime shared libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install ninja \
    && pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124 \
    && pip install "git+https://github.com/nianticlabs/spz"

WORKDIR /opt/re3draw
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
RUN pip install -e ".[dev]"

# Fail fast and loudly if the image cannot see a GPU, rather than silently training on the CPU.
ENV CUDA_VISIBLE_DEVICES=0
ENTRYPOINT ["re3draw-worker"]
CMD ["--help"]
