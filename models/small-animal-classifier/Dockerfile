FROM python:3.11-slim

# System dependencies for Pillow and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /opt/ml/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/requirements.txt \
    && find /usr/local/lib/python3.11/site-packages -name "*.pyc" -delete

# Create directories and download model weights from GitHub release
RUN mkdir -p /opt/ml/model /opt/ml/code /opt/ml/input /opt/ml/output && \
    wget -O /opt/ml/model/eva02-20260630-llrd.best.e02-s053514.stripped.pt \
    https://github.com/agentmorris/small-animal-classifier/releases/download/v1.0/eva02-20260630-llrd.best.e02-s053514.stripped.pt

# Copy model server code
COPY serve.py /opt/ml/code/serve.py

# Environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"
ENV MODEL_PATH=/opt/ml/model/eva02-20260630-llrd.best.e02-s053514.stripped.pt

WORKDIR /opt/ml/code
EXPOSE 8080
ENTRYPOINT ["python", "serve.py"]
