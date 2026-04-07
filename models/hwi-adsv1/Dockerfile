FROM python:3.8-slim

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
    && find /usr/local/lib/python3.8/site-packages -name "*.pyc" -delete

# Create directories and download model weights
RUN mkdir -p /opt/ml/model /opt/ml/code && \
    python -c "from huggingface_hub import hf_hub_download; \
        hf_hub_download(repo_id='Addax-Data-Science/HWI-ADS-v1', filename='always_crop_99710272_22x8_v12_epoch_00148.pt', local_dir='/opt/ml/model'); \
        hf_hub_download(repo_id='Addax-Data-Science/HWI-ADS-v1', filename='final-20260317.pt', local_dir='/opt/ml/model')"

# Copy model server code
COPY serve.py /opt/ml/code/serve.py

# Environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"
ENV BACKBONE_PATH=/opt/ml/model/always_crop_99710272_22x8_v12_epoch_00148.pt
ENV CHECKPOINT_PATH=/opt/ml/model/final-20260317.pt

WORKDIR /opt/ml/code
EXPOSE 8080
ENTRYPOINT ["python", "serve.py"]
