# MegaDetector v1000 Deployment

## Local Testing

### 1. Download the model weights

```bash
mkdir -p model-weights
cd model-weights
wget https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-redwood.pt
cd ..
```

### 2. Create the model archive

```bash
pip install torch-model-archiver

torch-model-archiver \
  --model-name mdv1000 \
  --version 1.0.0 \
  --handler mdv1000_handler.py \
  --extra-files model-weights/md_v1000.0.0-redwood.pt \
  --export-path model_store
```

This creates `model_store/mdv1000.mar` containing your handler code and the model weights.

### 3. Build the Docker container

```bash
docker build -t torchserve-mdv1000:0.5.3-cpu .
```

### 4. Run the container locally

```bash
bash docker_mdv1000.sh $(pwd)/model_store
```

### 5. Test the endpoint

```bash
curl http://127.0.0.1:8080/invocations -T /path/to/test/image.jpg
```

The response will be JSON with detections in Animl format.
