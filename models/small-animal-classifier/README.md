# Small Animal Classifier

Classification model for downward-facing small-animal camera traps. Trained and provided by [Dan Morris / agentmorris](https://github.com/agentmorris/small-animal-classifier).

## Model Information

- **Architecture**: EVA-02 Large (`eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`), via [timm](https://huggingface.co/docs/timm)
- **Classes**: 29 — `blank`, `setup_pickup`, `mouse`, `vole`, `rodent_other`, `woodrat_rat`, `squirrel`, `kangaroo_rat_pocket_mouse`, `chipmunk`, `pocket_gopher`, `rabbit_hare`, `shrew_mole`, `skunk`, `weasel`, `opossum`, `mammal_other`, `spiny_lizard`, `whiptail`, `snake`, `alligator_lizard`, `skink`, `rattlesnake`, `lizard_other`, `amphibian`, `bird`, `insect`, `isopod_crustacean`, `spider_arachnid`, `other_invert`
- **Input**: Full camera-trap frame (no bounding box crop)
- **Format**: ONNX (exported from PyTorch via `torch.onnx.export` with dynamo)
- **Runtime**: ONNX Runtime (CPU)
- **Source**: [github.com/agentmorris/small-animal-classifier](https://github.com/agentmorris/small-animal-classifier)

## Why ONNX?

The original PyTorch checkpoint (1.13 GB `.pt` file) requires `torch.load` + `timm.create_model` + `load_state_dict` to reconstruct the model at startup. On SageMaker Serverless Inference endpoints (6 GB RAM, 180s cold-start timeout), this deserialization process exceeded the timeout. EVA-02 Large has 304M parameters — the pickle deserialization + architecture rebuild is too slow on a constrained CPU instance.

Exporting to ONNX solves this because ONNX Runtime loads the model in a single C++ pass (protobuf read, no Python pickle, no architecture reconstruction). Load time drops from >180s to ~5-10s. As a bonus, we drop `torch`, `timm`, and `torchvision` from the container (~800 MB → ~19 MB for onnxruntime), reducing the container image from ~4 GB to ~250 MB.

The ONNX export produces numerically equivalent outputs — same weights, same operations. Predictions match to ~1e-5 tolerance (floating-point reordering in fused ops).

## How this model differs from other Animl classifiers

Most Animl classifiers (hwi-adsv1, nzi-adsv1, etc.) receive a MegaDetector bounding box and crop to the detected animal. This model operates on **full frames** from downward-facing cameras — there is no meaningful bbox crop stage. The animl-api interface sends only the image (no `bbox` field), and attaches a `[0, 0, 1, 1]` bbox to all returned predictions so they fit the standard detection schema.

## Preprocessing

Matches `src/transforms.py:ValTransform.__call__` from the training repo:

1. **Banner crop** — removes the camera's timestamp/temperature overlay (top ~3%, bottom ~3.5% of frame height). Fractions stored in the metadata JSON.
2. **Square resize** — squashes to 448×448 with BILINEAR interpolation. No aspect-ratio preservation. Camera faces down so there's no canonical orientation.
3. **Normalize** — standard ImageNet mean/std, read from metadata JSON.

Postprocessing is softmax over all 29 classes. No filtering; the animl-api caller applies confidence thresholds.

## Model Artifacts

Two files, stored in S3 and downloaded by SageMaker at endpoint startup via `ModelDataUrl`:

| File | Description | Location |
|------|-------------|----------|
| `small-animal-classifier.onnx` + `.onnx.data` | ONNX exported model (~1.1 GB) | `s3://sagemaker-us-west-2-830244800171/small-animal-classifier/` |
| `small-animal-classifier-metadata.json` | Preprocessing config (class names, img_size, norm stats, banner crop fractions) | Same S3 path |

These are packaged as `small-animal-classifier.tar.gz` for SageMaker's `ModelDataUrl` requirement. SageMaker extracts them into `/opt/ml/model/` at startup.

The ONNX export is performed via `onnx_conversion.ipynb` in this directory using `torch.onnx.export(..., dynamo=True)`. The export requires `torch`, `timm`, `onnx`, and `onnxscript` — these are dev-time dependencies only, not needed in the deployed container.

## Local Development

### Build

```bash
docker buildx build --platform linux/amd64 -t small-animal-classifier .
```

### Run

Model weights are not baked into the image. Mount them locally:

```bash
docker run -p 8080:8080 -v $(pwd)/model-weights:/opt/ml/model small-animal-classifier
```

The `model-weights/` directory must contain `small-animal-classifier.onnx`, `small-animal-classifier.onnx.data`, and `small-animal-classifier-metadata.json`.

### Test

```bash
curl http://localhost:8080/ping
# {"status":"Healthy"}

curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64-encoded-image>"}'
```

## Request / Response Format

**Request** — JSON with a single field:
```json
{"image": "iVBORw0KGgoAAAANSUhEUgAA..."}
```

No `bbox` field. The model processes the full frame.

**Response** — all 29 classes with softmax confidence scores:
```json
{
  "blank": 0.001,
  "mouse": 0.542,
  "squirrel": 0.004,
  ...
}
```

## Deployment

Deployment follows the standard Animl pattern. See `small-animal-classifier_deploy.ipynb`.

Key differences from other models:
- Model artifacts are loaded from S3 via `ModelDataUrl` (tar.gz), not baked into the container image
- The `create_model` call includes `ModelDataUrl` pointing to the tar.gz in `s3://sagemaker-us-west-2-830244800171/small-animal-classifier/`
- Container image is ~250 MB (no torch/timm), so cold starts are fast

### SSM Parameters

| Parameter Name | Parameter Value |
|---|---|
| `/ml/small-animal-classifier-batch-endpoint-dev` | `small-animal-classifier-concurrency-20` |
| `/ml/small-animal-classifier-batch-endpoint-prod` | `small-animal-classifier-concurrency-20` |
| `/ml/small-animal-classifier-realtime-endpoint-dev` | `small-animal-classifier-concurrency-20` |
| `/ml/small-animal-classifier-realtime-endpoint-prod` | `small-animal-classifier-concurrency-20` |

### animl-api integration

The model interface is in `animl-api/src/ml/modelInterfaces.ts` registered as `'small-animal-classifier'`. It sends only `{"image": "<base64>"}` (no bbox) and attaches `[0, 0, 1, 1]` to all returned predictions.

## Validation

`validation/local_validation.ipynb` compares predictions from the deployed container against reference output from the training repo. Predictions match to 4 decimal places (5th-6th decimal differences are expected due to ONNX Runtime's operator fusion).
