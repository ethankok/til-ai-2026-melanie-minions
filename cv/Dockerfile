# Dockerfile for building the CV image.


# The base image, an example deep learning VM.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV CV_MODEL_PATH=/workspace/models/cv/best.pt
ENV CV_CATEGORY_MAP='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]'
ENV CV_CONF=0.25
ENV CV_IOU=0.50
ENV CV_IMGSZ=768
WORKDIR /workspace

# Installs your dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies your model weights and source files.
# Put the trained checkpoint at cv/models/best.pt before building/submitting.
COPY models /workspace/models/cv
COPY src .

# Starts your model server.
CMD uvicorn cv_server:app --port 5002 --host 0.0.0.0
