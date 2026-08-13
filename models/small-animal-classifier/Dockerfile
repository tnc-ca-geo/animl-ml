FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /opt/ml/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/requirements.txt \
    && find /usr/local/lib/python3.10/site-packages -name "*.pyc" -delete

# Create directories and download model weights
RUN mkdir -p /opt/ml/model /opt/ml/code

# Download model checkpoint from the releases page
RUN wget https://github.com/agentmorris/small-animal-classifier/releases/download/v1.0/best_stripped.ckpt -O /opt/ml/model/best_stripped.ckpt

# Copy model server code
COPY serve.py /opt/ml/code/serve.py

# Environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"
ENV CHECKPOINT_PATH=/opt/ml/model/best_stripped.ckpt

WORKDIR /opt/ml/code
EXPOSE 8080
ENTRYPOINT ["python", "serve.py"]