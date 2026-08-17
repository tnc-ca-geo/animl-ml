FROM python:3.13-slim

# System dependencies for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /opt/ml/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/requirements.txt \
    && find /usr/local/lib/python3.13/site-packages -name "*.pyc" -delete

# Create directories and copy model files
# NOTE: For local testing, mount model-weights/ as a volume.
# For production, replace these lines with wget from your hosting location
# (GitHub release, S3, etc.)
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
