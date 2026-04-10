# HWI-ADS-v1 SageMaker Handler

Credit to [Peter van Lunteren of Addax Data Science](https://addaxdatascience.com/) for training and open sourcing this classifier, and to the USDA Forest Service (IPIF) & The Nature Conservancy (TNC) for funding the work.

This handler serves the HWI-ADS-v1 (Hawaiian Wildlife Invasives) model for inference via FastAPI.

## Model Information

- **Model**: Hawaiian Wildlife Invasives v1
- **Framework**: PyTorch (SpeciesNet EfficientNet V2 M backbone + fine-tuned linear head)
- **Input**: 480x480 RGB images (cropped to detection, then resized)
- **Classes**: 15 (see taxon-mapping.csv)
- **Developer**: Addax Data Science
- **Owner**: USDA Forest Service – Pacific Southwest Research Station, Institute of Pacific Islands Forestry (IPIF) & The Nature Conservancy (TNC)
- **License**: CC BY-NC 4.0
- **HuggingFace**: [Addax-Data-Science/HWI-ADS-v1](https://huggingface.co/Addax-Data-Science/HWI-ADS-v1)

## Architecture

The model is a two-part architecture built on Google's [SpeciesNet](https://github.com/google/cameratrapai) framework:

1. **Backbone**: SpeciesNet's EfficientNet V2 M classifier (v4.0.2a, the "always-crop" variant), converted from ONNX to PyTorch via `onnx2torch`. This is the frozen feature extractor — it outputs a flat 2498-dimensional feature vector (the onnx2torch conversion includes global average pooling within the backbone). Weights file: `always_crop_99710272_22x8_v12_epoch_00148.pt` (224MB).
2. **Head**: A single `nn.Linear(2498, 15)` layer fine-tuned on Hawaiian camera trap data. The head input size (2498) is read from the checkpoint's trained weights at load time. Stored in a checkpoint dict alongside metadata. Weights file: `final-20260317.pt` (224MB).

The backbone expects NHWC input layout (from its TensorFlow/ONNX origin). Our forward pass converts from PyTorch's native NCHW format.

## Key Implementation Decisions

All preprocessing and inference decisions are based on the reference implementation by the model developer:

### Cropping: Simple tight crop (no padding or squaring)

Source: [`get_crop` in AddaxAI classify_detections.py, lines 218-235](https://github.com/PetervanLunteren/AddaxAI/blob/main/classification_utils/model_types/addax-sppnet/classify_detections.py)

We use a simple tight crop from the MegaDetector bounding box with no padding, squaring, or letterboxing. This differs from some other Animl models (e.g., nzi-adsv1 uses Dan Morris's padded square crop). We match the training-time crop to avoid distribution shift.

### Preprocessing: Resize → ToTensor → Normalize (no aspect ratio preservation)

Source: [`preprocess` in AddaxAI classify_detections.py, lines 168-172](https://github.com/PetervanLunteren/AddaxAI/blob/main/classification_utils/model_types/addax-sppnet/classify_detections.py)

The crop is resized directly to 480x480 (stretching if non-square). Mean/std normalization values are read from the checkpoint. No letterboxing is applied.

### Postprocessing: Softmax

Source: [`get_classification` in AddaxAI classify_detections.py, lines 181-200](https://github.com/PetervanLunteren/AddaxAI/blob/main/classification_utils/model_types/addax-sppnet/classify_detections.py)

Raw logits are converted to probabilities via `F.softmax(output, dim=1)`. This is a single-label classifier (probabilities sum to 1.0), unlike multi-label models that use sigmoid.

### Model architecture: FXClassifier

Source: [`FXClassifier` class in AddaxAI classify_detections.py, lines 72-130](https://github.com/PetervanLunteren/AddaxAI/blob/main/classification_utils/model_types/addax-sppnet/classify_detections.py)

The wrapper class and backbone loading function are adapted from the AddaxAI reference, simplified for our single known backbone. The backbone outputs a flat 2D tensor `[batch, 2498]` (the onnx2torch conversion bakes global average pooling into the backbone), so we use a simple `flatten(1)` rather than the generic ndim branching in the original code. The head input size is read from the checkpoint's `head.weight` shape at load time rather than hardcoded.

### Backbone identity: EfficientNet V2 M

Source: [SpeciesNet README](https://github.com/google/cameratrapai) — "trained at Google using a large dataset of camera trap images and an EfficientNet V2 M architecture"

## Docker Build and Run

### Build the Docker image
```bash
docker buildx build --platform linux/amd64 -t hwi-adsv1 .
```

### Run locally
```bash
docker run -p 8080:8080 hwi-adsv1
```

### Test the endpoint
```bash
# Health check
curl http://localhost:8080/ping

# Run inference
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64-encoded-image>",
    "bbox": [0.1, 0.2, 0.5, 0.6]
  }'
```

## Request Format

The handler expects a JSON payload with:
- `image`: Base64-encoded image string
- `bbox`: Bounding box in [x, y, width, height] format (normalized 0-1)

```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAA...",
  "bbox": [0.1, 0.2, 0.5, 0.6]
}
```

## Output Format

Returns a JSON object mapping class names to confidence scores:

```json
{
    "bird": 0.001,
    "rodent": 0.002,
    "mongoose": 0.95,
    "cat": 0.003,
    ...
}
```

All 15 classes are returned with their confidence scores (no filtering applied).

## Implementation Q&A

Questions and answers that came up during implementation, preserved for future reference.

### Why two weight files instead of one?

Unlike the NZI-ADS-v1 model (a single YOLOv8 `.pt` file), this model has two files because they come from two different training processes. The backbone (`always_crop_...pt`) was trained by Google as part of SpeciesNet. The checkpoint (`final-20260317.pt`) contains the head that Addax Data Science fine-tuned on Hawaiian wildlife data. The backbone is frozen (unchanged) — only the head was trained.

### Why does the AddaxAI example try two different backbone files?

The example code tries `always_crop_...pt` first, then falls back to `full_image_...pt`. These are SpeciesNet's two classifier variants (v4.0.2a and v4.0.2b). Since the HuggingFace repo for HWI-ADS-v1 only includes the `always_crop` backbone, we don't need the fallback logic. We also simplified the `FXClassifier` class — the generic version used a dummy input trick to determine the backbone output size and had ndim branching for different output shapes. We instead read `in_features` from the checkpoint's `head.weight` shape (2498) and use a simple `flatten(1)` since we confirmed at runtime that the backbone outputs a flat 2D tensor.

### Why is there a Windows path compatibility hack?

The `pathlib.WindowsPath = pathlib.PosixPath` line appears in both the [AddaxAI reference code](https://github.com/PetervanLunteren/AddaxAI/blob/main/classification_utils/model_types/addax-sppnet/classify_detections.py) and our [nzi-adsv1 serve.py](/models/nzi-adsv1/serve.py). When a model is saved on Windows, Python's pickle serializer can embed Windows-style path objects. This alias prevents deserialization failures on Unix/Mac. We don't know for certain the model was saved on Windows, but the Addax team included this in their own inference code, so we keep it as a defensive measure. It's a no-op if the paths aren't Windows-style.

### Why use a simple tight crop instead of Dan Morris's padded square crop?

The model was trained on simple tight crops (no padding, no squaring). Using a different crop method — even a "smarter" one — would introduce distribution shift: the model would see input that looks different from what it learned on. The crop strategy is part of the model's implicit contract. If you wanted to use a different crop, you'd need to retrain the head with that crop method. Ref: [`get_crop` in AddaxAI classify_detections.py](https://github.com/PetervanLunteren/AddaxAI/blob/main/classification_utils/model_types/addax-sppnet/classify_detections.py)

### Why no letterboxing?

Same reasoning as the crop method. The training pipeline uses `transforms.Resize` which stretches the crop to 480x480 without preserving aspect ratio. Other Animl models (e.g., nzi-adsv1) use letterboxing because that's how *those* models were trained. Always match the training pipeline for preprocessing.

### Does the image get recompressed during processing?

No. We decode the base64 string back to the original bytes, PIL opens those bytes directly, cropping happens in pixel space, and resize happens during tensor conversion. At no point do we re-encode to JPEG. The pixel data going into the model is as faithful to the original as possible (minus resize interpolation, which is unavoidable).

### Why `weights_only=True` with a fallback to `weights_only=False`?

`weights_only=True` is a PyTorch security feature that restricts pickle deserialization to only tensor data. The fallback to `weights_only=False` is needed when files contain custom objects (like the backbone's `onnx2torch` classes). It's not error handling — it's a compatibility pattern: try the safe way first, fall back to the permissive way if needed.

### Why does the backbone need NHWC layout?

The SpeciesNet backbone was originally a TensorFlow model exported to ONNX, then converted to PyTorch via `onnx2torch`. TensorFlow uses NHWC (batch, height, width, channels) natively, while PyTorch uses NCHW (batch, channels, height, width). The `permute` call in the forward pass rearranges dimensions to match what the backbone expects.

### Why `onnx2torch` as a dependency?

The backbone file contains serialized `onnx2torch` custom op classes (visible in the [HuggingFace pickle scan](https://huggingface.co/Addax-Data-Science/HWI-ADS-v1/blob/main/always_crop_99710272_22x8_v12_epoch_00148.pt)): `OnnxPadStatic`, `OnnxBinaryMathOperation`, `OnnxReshape`, `OnnxSqueezeDynamicAxes`, `OnnxGlobalAveragePoolWithKnownInputShape`, `OnnxMatMul`, `OnnxTranspose`. These classes must be importable for `torch.load()` to deserialize the backbone.
