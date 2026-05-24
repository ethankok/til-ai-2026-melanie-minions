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
ENV CV_MODEL_FAMILY=auto
ENV CV_CATEGORY_MAP='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]'
ENV CV_CONF=0.20
ENV CV_IOU=0.55
ENV CV_IMGSZ=1280
ENV CV_MAX_DET=100
ENV CV_AUGMENT=1
ENV CV_HALF=1
ENV CV_CROSS_CLASS_NMS_IOU=0.97
ENV CV_SECOND_PASS=0
ENV CV_SECOND_CONF=0.15
ENV CV_SECOND_IOU=0.55
ENV CV_SECOND_IMGSZ=896
ENV CV_SECOND_AUGMENT=1
ENV CV_SECOND_SCORE_SCALE=0.20
ENV CV_SECOND_MERGE_IOU=0.95
ENV CV_SECOND_MIN_DETECTIONS=7
ENV CV_OWLV2_MODEL_ID=google/owlv2-base-patch16-ensemble
ENV CV_OWLV2_NMS_IOU=0.50
ENV CV_OWLV2_LOCAL_FILES_ONLY=1
WORKDIR /workspace

# Installs your dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ARG CV_PREFETCH_OWLV2=0
RUN if [ "$CV_PREFETCH_OWLV2" = "1" ]; then \
        python -c "import os; from transformers import Owlv2ForObjectDetection, Owlv2Processor; m=os.environ.get('CV_OWLV2_MODEL_ID', 'google/owlv2-base-patch16-ensemble'); Owlv2Processor.from_pretrained(m); Owlv2ForObjectDetection.from_pretrained(m)"; \
    fi

# Copies your model weights and source files.
# Put the trained checkpoint at cv/models/best.pt before building/submitting.
COPY models /workspace/models/cv
COPY src .

# Starts your model server.
CMD uvicorn cv_server:app --port 5002 --host 0.0.0.0
