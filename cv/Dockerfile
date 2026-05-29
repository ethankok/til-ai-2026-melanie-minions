# Dockerfile for building the CV image.
#
# CURRENTLY CONFIGURED FOR THE RF-DETR EXPERIMENT BUILD (CV_MODEL_FAMILY=rfdetr)
# — the active architecture-family bet, see cv/NOTES.md "Active line". The
# shipped YOLO11l champion ENV is preserved in the commented block below; to
# build the champion instead, comment the RF-DETR block and uncomment the
# champion block (or `git checkout main -- cv/Dockerfile`).


# The base image, an example deep learning VM.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

# --- RF-DETR experiment serving config (ACTIVE) ---
# Serve checkpoint_best_ema.pth copied to cv/models/best.pth. CV_IOU/CV_IMGSZ are
# ignored on the rfdetr path (NMS-free, resolution fixed at load); CV_CONF is the
# predict threshold. CV_CATEGORY_MAP undoes the offset-1 category ids the
# RF-DETR dataset builder used (predicts 1..18 -> TIL 0..17).
ENV CV_MODEL_PATH=/workspace/models/cv/best.pth
ENV CV_MODEL_FAMILY=rfdetr
ENV CV_CATEGORY_MAP='{"1":0,"2":1,"3":2,"4":3,"5":4,"6":5,"7":6,"8":7,"9":8,"10":9,"11":10,"12":11,"13":12,"14":13,"15":14,"16":15,"17":16,"18":17}'
ENV CV_CONF=0.30
ENV CV_RFDETR_RESOLUTION=728
ENV CV_RFDETR_VARIANT=base
ENV CV_AUGMENT=0
ENV CV_CROSS_CLASS_NMS_IOU=0

# --- Shipped YOLO11l champion config (0.671/0.950, blended 0.7410) ---
# Uncomment this block (and comment the RF-DETR block above) to build the
# champion. Requires the YOLO11l weights at cv/models/best.pt.
# ENV CV_MODEL_PATH=/workspace/models/cv/best.pt
# ENV CV_MODEL_FAMILY=auto
# ENV CV_CATEGORY_MAP='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]'
# ENV CV_CONF=0.20
# ENV CV_IOU=0.55
# ENV CV_IMGSZ=1280
# ENV CV_MAX_DET=100
# ENV CV_AUGMENT=1
# ENV CV_HALF=1
# ENV CV_CROSS_CLASS_NMS_IOU=0.97
# ENV CV_SECOND_PASS=0

WORKDIR /workspace

# Installs your dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Offline prefetch: bake RF-DETR's base + DINOv2 backbone weights into the image
# while the build still has network, so the no-internet eval network never
# triggers a download when the server constructs the model at startup. Forced to
# CPU (no GPU at build time). The TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE ENVs are
# set AFTER this RUN so they only take effect at runtime, not during the
# prefetch download. NOT masked with `|| true`: if this fails, the image would
# silently 0-score on the offline eval network, so we want the build to fail
# loudly instead. (Harmless extra download when building the YOLO champion.)
RUN CUDA_VISIBLE_DEVICES="" python -c "from rfdetr import RFDETRBase; RFDETRBase(resolution=728)"
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Copies your model weights and source files.
# Put the trained checkpoint at cv/models/best.pth (RF-DETR) or best.pt (YOLO)
# before building/submitting.
COPY models /workspace/models/cv
COPY src .

# Starts your model server.
CMD uvicorn cv_server:app --port 5002 --host 0.0.0.0
