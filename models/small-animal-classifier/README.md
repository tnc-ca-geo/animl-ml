# Small Animal Classifier

This handler serves the small-animal-classifier model for inference via FastAPI.

## Model Information

- **Model**: California Small Animals Classifier
- **Framework**: PyTorch (timm — EVA-02 Large, 448px)
- **Input**: Full camera-trap frame (no bbox crop required)
- **Classes**: 29 (see `SINGLE_IMAGE_INFERENCE.md` in the training repo)
- **Developer**: [Dan Morris / agentmorris](https://github.com/agentmorris)
- **Source repo**: [github.com/agentmorris/small-animal-classifier](https://github.com/agentmorris/small-animal-classifier)

### Class list

`blank`, `setup_pickup`, `mouse`, `vole`, `rodent_other`, `woodrat_rat`, `squirrel`,
`kangaroo_rat_pocket_mouse`, `chipmunk`, `pocket_gopher`, `rabbit_hare`, `shrew_mole`,
`skunk`, `weasel`, `opossum`, `mammal_other`, `spiny_lizard`, `whiptail`, `snake`,
`alligator_lizard`, `skink`, `rattlesnake`, `lizard_other`, `amphibian`, `bird`,
`insect`, `isopod_crustacean`, `spider_arachnid`, `other_invert`

## Architecture

The model is a single [timm](https://huggingface.co/docs/timm) model
(`eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`) rebuilt from a stripped
inference checkpoint (`.stripped.pt`). The checkpoint stores the architecture
name, class list, input size, normalization stats, and banner-crop fractions
alongside the model weights, so the handler is fully self-contained once the
checkpoint file is present.

## Key Implementation Decisions

### Why no bbox crop?

Unlike other Animl classifiers (e.g., `hwi-adsv1`, `nzi-adsv1`) that receive a
MegaDetector bounding box and crop to the detected animal, this model was
trained on **full frames** from downward-facing small-animal cameras. There is
no meaningful bbox crop stage — the whole image is the input. The animl-api
interface for this model sends only the image, not a bbox.

### Preprocessing: banner crop → square resize → normalize

Ref: `transforms.py::ValTransform` in the training repo.

1. **Banner crop**: Strips the camera's timestamp/temperature overlay band
   (top ≈ 3%, bottom ≈ 3.5% of frame height). Values are read from the
   checkpoint's `banner_crop` dict. This prevents the model from using
   metadata text as a classification shortcut.
2. **Square resize**: Squashes the cropped frame to `img_size × img_size`
   (448×448) using `BILINEAR` interpolation, without preserving aspect ratio.
   Because the camera faces downward, there is no canonical orientation once
   the banner is removed, so squashing is acceptable.
3. **ToTensor + Normalize**: Standard ImageNet normalization using mean/std
   read from the checkpoint.

### Postprocessing: softmax over all classes

Single-label classification → `torch.softmax(logits.float(), dim=1)`.
All 29 class scores are returned; filtering to a confidence threshold is done
by the animl-api caller via `_filterClassifierPredictions`.

## Docker Build and Run

### Build

```bash
docker buildx build --platform linux/amd64 -t small-animal-classifier .
```

### Run locally (with weights already downloaded to ./model-weights/)

```bash
docker run -p 8080:8080 \
  -e MODEL_PATH=/opt/ml/model/model.stripped.pt \
  -v $(pwd)/model-weights:/opt/ml/model \
  small-animal-classifier
```

### Test

```bash
# Health check
curl http://localhost:8080/ping

# Inference
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64-encoded-image>"}'
```

## Request Format

```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

- `image`: Base64-encoded image (JPEG, PNG, etc.)

## Output Format

```json
{
  "blank": 0.001,
  "mouse": 0.002,
  "squirrel": 0.931,
  "chipmunk": 0.044,
  ...
}
```

All 29 classes are returned with their softmax confidence scores. No filtering
is applied; the animl-api caller handles thresholding.

## Deployment to SageMaker

See `deploy_to_sagemaker.ipynb` (copy from an existing model such as `nzi-adsv1`
and adjust the endpoint name, concurrency, and container URI).

### SSM Parameters to create

| Parameter Name | Parameter Value |
|---|---|
| `/ml/small-animal-classifier-batch-endpoint-dev` | `small-animal-classifier-concurrency-<N>` |
| `/ml/small-animal-classifier-batch-endpoint-prod` | `small-animal-classifier-concurrency-<N>` |
| `/ml/small-animal-classifier-realtime-endpoint-dev` | `small-animal-classifier-concurrency-<N>` |
| `/ml/small-animal-classifier-realtime-endpoint-prod` | `small-animal-classifier-concurrency-<N>` |
