FROM python:3.13-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /opt/ml/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/requirements.txt \
    && find /usr/local/lib/python3.13/site-packages -name "*.pyc" -delete

# Create directories and download model weights
RUN mkdir -p /opt/ml/model /opt/ml/code /opt/ml/input /opt/ml/output && \
    wget -O /opt/ml/model/md_v1000.0.0-redwood.pt \
    https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-redwood.pt

# Copy model server code
COPY serve.py /opt/ml/code/serve.py

# Environment variables for SageMaker
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"
ENV MODEL_PATH=/opt/ml/model/md_v1000.0.0-redwood.pt
ENV DETECTION_THRESHOLD=0.0005

# Set working directory
WORKDIR /opt/ml/code

# Expose the port that the FastAPI server will run on
EXPOSE 8080

# Entry point for SageMaker
ENTRYPOINT ["python", "serve.py"]
