FROM python:3.13-slim

# System dependencies for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /opt/ml/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/requirements.txt \
    && find /usr/local/lib/python3.13/site-packages -name "*.pyc" -delete

# Create directories required by SageMaker
# NOTE: Model artifacts are NOT baked into this image. They are downloaded by
# SageMaker at endpoint startup from s3://animl-model-zoo/small-animal-classifier/
# and mounted at /opt/ml/model/. The following files must exist in that S3 path:
#   - small-animal-classifier.onnx          (ONNX exported model, ~1.1 GB)
#   - small-animal-classifier-metadata.json  (preprocessing config, ~1 KB)
#
# For local testing, mount a directory containing these files as a volume:
#   docker run -p 8080:8080 -v $(pwd)/model-weights:/opt/ml/model small-animal-classifier
RUN mkdir -p /opt/ml/model /opt/ml/code /opt/ml/input /opt/ml/output

# Copy model server code
COPY serve.py /opt/ml/code/serve.py

# Environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"
ENV MODEL_PATH=/opt/ml/model/small-animal-classifier.onnx
ENV METADATA_PATH=/opt/ml/model/small-animal-classifier-metadata.json

WORKDIR /opt/ml/code
EXPOSE 8080
ENTRYPOINT ["python", "serve.py"]
