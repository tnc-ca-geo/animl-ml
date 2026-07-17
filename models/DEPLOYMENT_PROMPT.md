# Reusable Prompt: Deploying a New Classification Model to Animl via SageMaker

Use this prompt when deploying a new classification model. Fill in the blanks and provide it to an AI assistant or use it as a checklist.

---

## The Prompt

```
I need to deploy a new classification model to Animl's SageMaker Serverless Inference infrastructure. Here is the information about the model:

- Model name: [e.g., hwi-adsv1]
- HuggingFace / model source: [URL]
- Reference inference code (if available): [URL to example showing how to load and run the model]
- Framework: [PyTorch / TensorFlow / PyTorch Lightning / YOLO / SpeciesNet FXClassifier / other]
- Owner / developer: [who trained it, who funded it]
- License: [e.g., CC BY-NC 4.0]

The serve.py should follow the Animl conventions:
- FastAPI app with GET /ping and POST /invocations endpoints
- /invocations accepts JSON with "image" (base64-encoded) and "bbox" ([x, y, w, h] normalized 0-1)
- Returns a dict of {class_name: confidence_score} for ALL classes (no filtering)
- Model loads at container startup (module level), not per-request
- Weights stored in /opt/ml/model/ with env var overrides for local testing
- Logging with timing for model load and each request
- Container crashes on startup if model fails to load

Existing model implementations to reference are in the /models directory of the animl-ml repo.
```

---

## What You Need to Determine for Each Model

These are the key questions that vary per model. The answers drive every implementation decision in serve.py.

### 1. How is the model loaded?

Different frameworks load models differently:

| Framework | Loading pattern | Example model |
|-----------|----------------|---------------|
| YOLO (ultralytics) | `model = YOLO(path)` — one file, handles everything | nzi-adsv1 |
| SpeciesNet FXClassifier | Two files: backbone (.pt GraphModule) + checkpoint (.pt dict). Assemble with `FXClassifier(backbone, ...)` then `load_state_dict()` | hwi-adsv1 |
| TorchServe | Handler class extending `ImageClassifier`, loaded by TorchServe runtime | alitav3, sdzwa-southwestv3 |
| TensorFlow | SavedModel bundle loaded by TF Serving, wrapped in FastAPI | irc |

For FXClassifier models specifically:
- The backbone is a SpeciesNet EfficientNet V2 M converted from ONNX via onnx2torch
- It's serialized as a torch.fx.GraphModule — needs `add_safe_globals([reduce_graph_module])` to deserialize
- The checkpoint dict contains: `model` (state_dict), `num_classes`, `img_size`, `input_layout`, `class_names`, `normalize` (mean/std)
- Requires `onnx2torch` as a dependency for the backbone's custom op classes

### 2. What preprocessing does the model expect?

**Always match the training pipeline.** Check the reference inference code for:

- **Crop method**: How is the MegaDetector bbox used to extract the animal?
  - Simple tight crop (hwi-adsv1) — just crop to bbox pixel coords
  - Dan Morris padded square crop (nzi-adsv1) — square the box, add padding, pad with black
  - Tightest square crop (alitav3) — square the box, pad with black, no extra padding
  - Each model has its own crop method. Using the wrong one hurts accuracy.

- **Resize**: What input size? Is aspect ratio preserved (letterboxing) or stretched?
  - 480x480 stretched (hwi-adsv1, FXClassifier models)
  - 224x224 with CenterCrop (nzi-adsv1, YOLO handles internally)
  - 480x480 stretched (alitav3)

- **Normalization**: What mean/std values?
  - Read from checkpoint dict (FXClassifier models)
  - Handled internally (YOLO models)
  - ImageNet defaults [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] (alitav3)

- **Bbox format**: What format does the crop function expect?
  - [x, y, width, height] normalized (MegaDetector format, used by Animl)
  - [ymin, xmin, ymax, xmax] normalized (alitav3 — needs conversion)

### 3. What postprocessing converts raw output to probabilities?

| Method | When to use | Example |
|--------|-------------|---------|
| Softmax | Single-label classification (probabilities sum to 1.0) | hwi-adsv1 |
| Sigmoid | Multi-label classification (each class independent) | alitav3 |
| Built-in | Framework handles it | nzi-adsv1 (YOLO) |

### 4. What dependencies are needed?

Check the model's pickle imports (visible on HuggingFace file pages) and the reference code imports. Common patterns:

| Model type | Key dependencies |
|------------|-----------------|
| YOLO | ultralytics, torch, torchvision, dill |
| FXClassifier | torch, torchvision, onnx2torch |
| TorchServe | torch, torchvision, torchserve |
| TensorFlow | tensorflow, fastapi |

All models also need: pillow, fastapi, uvicorn, huggingface-hub (if downloading weights in Dockerfile).

---

## serve.py Structure

Every serve.py follows this structure regardless of model type:

```
1. Imports and setup
   - Standard lib: io, base64, os, time, logging, json
   - PIL for image handling
   - FastAPI + uvicorn for the web server
   - Framework-specific imports (torch, ultralytics, tensorflow, etc.)
   - ImageFile.LOAD_TRUNCATED_IMAGES = True
   - Windows path compat hack (for Addax models)

2. Model architecture definition (if needed)
   - YOLO: not needed, ultralytics handles it
   - FXClassifier: load_fx_checkpoint() function + FXClassifier class
   - TorchServe: handler class

3. Model loading (runs once at startup, module level)
   - Load weights from /opt/ml/model/ with env var overrides
   - Assemble model, load state dict, set to eval mode
   - Build preprocessing pipeline (transforms.Compose or equivalent)
   - Extract class names
   - Log load time and class count
   - Crash on failure (raise, don't catch)

4. FastAPI endpoints
   - GET /ping → {"status": "Healthy"}
   - POST /invocations:
     a. Decode base64 image → PIL Image → ensure RGB
     b. Crop using model-specific crop function
     c. Preprocess (resize, to tensor, normalize)
     d. Add batch dimension (.unsqueeze(0))
     e. Forward pass with torch.no_grad()
     f. Postprocess (softmax or sigmoid)
     g. Format as {class_name: score} dict
     h. Log timing, return predictions

5. Main entrypoint
   - uvicorn.run(app, host="0.0.0.0", port=8080)
```

---

## Checklist Before Deploying

- [ ] Identified model framework and loading pattern
- [ ] Found reference inference code from model developer
- [ ] Determined crop method (match training pipeline exactly)
- [ ] Determined resize strategy (stretched vs letterboxed, input size)
- [ ] Determined normalization values (mean/std)
- [ ] Determined postprocessing (softmax vs sigmoid vs built-in)
- [ ] Identified all dependencies (check pickle imports on HuggingFace)
- [ ] Created serve.py following the structure above
- [ ] Created requirements.txt
- [ ] Created Dockerfile
- [ ] Tested container locally (docker build, docker run, curl /ping, curl /invocations)
- [ ] Created deployment notebook (copy from similar model)
- [ ] Created SSM parameters (batch + realtime, dev + prod)
- [ ] Updated animl-api (SSM params + model interface)
- [ ] Created MLModel record in MongoDB
- [ ] Added model to Project.availableMLModels
