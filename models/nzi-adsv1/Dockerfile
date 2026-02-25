FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages with exact versions from model training
COPY requirements.txt /opt/ml/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/requirements.txt \
    && find /usr/local/lib/python3.8/site-packages -name "*.pyc" -delete

# Create directories and download model weights
RUN mkdir -p /opt/ml/model /opt/ml/code /opt/ml/input /opt/ml/output && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Addax-Data-Science/NZI-ADS-v1', filename='new_zealand_v1.pt', local_dir='/opt/ml/model')"

# Copy model server code
COPY serve.py /opt/ml/code/serve.py

# Environment variables for SageMaker
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"
ENV MODEL_PATH=/opt/ml/model/new_zealand_v1.pt

# Set working directory
WORKDIR /opt/ml/code

# Expose the port that the FastAPI server will run on
EXPOSE 8080

# Entry point for SageMaker
ENTRYPOINT ["python", "serve.py"]
