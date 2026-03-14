FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install \
    autodistill \
    autodistill-grounding-dino \
    supervision \
    roboflow \
    scikit-learn \
    opencv-python \
    numpy \
    pillow \
    tqdm

RUN pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

CMD ["bash"]