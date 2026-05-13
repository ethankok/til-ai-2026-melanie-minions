# Dockerfile for building the CV image.


# The base image, an example deep learning VM.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV CV_MODEL_PATH=yolov8n.pt
WORKDIR /workspace

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache the default pretrained detector during image build so runtime inference
# does not depend on network access.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copies your source files.
COPY src .

# Starts your model server.
CMD uvicorn cv_server:app --port 5002 --host 0.0.0.0
