# serve.py Implementation Review

Overall verdict: the implementation is **mostly correct** but has several meaningful issues worth fixing.

---

## Model Loading ✅ with one bug

The checkpoint loading logic is structurally correct — reads from `ck["classes"]`, `ck["model_name"]`, `ck["img_size"]`, `ck["norm_mean"]`, `ck["norm_std"]`, `ck["banner_crop"]` — all of which match the fields written by `strip_checkpoint.py`.

**Bug**: `torch.load(..., weights_only=True)` will fail at runtime. The reference (`run_inference.py`, `strip_checkpoint.py`, `SINGLE_IMAGE_INFERENCE.md`) all use `weights_only=False`. The stripped checkpoint contains a plain dict with standard Python types and torch tensors, but `weights_only=True` is still stricter than necessary and the reference code explicitly avoids it. The hwi-adsv1 handler uses a try/fallback pattern (`weights_only=True` → `weights_only=False`) as a safe approach; this handler should do the same or just use `weights_only=False` directly.

**Minor**: `banner_top` and `banner_bot` are extracted in `load_model()` but never returned or stored anywhere — they're read and immediately discarded. More on this below.

---

## Preprocessing ⚠️ Two issues

**Issue 1 — Resize interpolation mismatch (correctness bug):** The `ValTransform` in `transforms.py` uses `img.resize((s, s), Image.BILINEAR)` — a PIL resize with bilinear interpolation applied *before* tensor conversion. The handler instead uses `transforms.Resize((img_size, img_size), antialias=True)` from torchvision, which resizes as a tensor operation (after `ToTensor`) with a different interpolation path. These are not numerically equivalent. The correct approach is to either use PIL's `resize` directly as in `ValTransform.__call__`, or replicate the same sequence: PIL BILINEAR resize → `TF.to_tensor` → `TF.normalize`.

**Issue 2 — Banner crop fractions are hardcoded instead of using checkpoint values:** The handler calls `crop_banner(image, 0.03, 0.035)` with hardcoded literals, even though it reads `banner_top`/`banner_bot` from the checkpoint in `load_model()` — but then throws them away. The correct thing is to use the values from the checkpoint (`ck["banner_crop"]["top"]` and `ck["banner_crop"]["bottom"]`), which happen to currently be 0.03/0.035, but could change if the model is retrained. The reference's `load_model()` passes them directly into `ValTransform`.

The `crop_banner` function logic itself is a correct copy from `transforms.py`.

---

## Inference ✅ Correct

`torch.no_grad()`, `softmax(output, dim=1)`, `unsqueeze(0)` for batch dim — all correct and match the reference.

---

## Output Format ❌ Wrong for Animl's conventions

This is the most significant problem. The output format is based on the *reference repo's batch output format* rather than the Animl handler convention. Specifically:

**Wrong: returns all classes as index-keyed pairs**
```python
confidences = []
for idx, score in enumerate(scores):
    confidences.append([str(idx), round(score, 4)])
```
This returns all 29 classes as `[["0", 0.0012], ["1", 0.9311], ...]` — integer-index strings rather than class names.

**Wrong: wraps in a `{"images": [...]}` top-level structure**  
Every other Animl handler (hwi-adsv1, nzi-adsv1, alitav3, etc.) returns a flat `{class_name: confidence}` dict directly. The `images` wrapper is the batch file output format from `run_inference.py`, not the per-request API format.

**Wrong: returns only top-N (3) not all classes**  
The reference batch inference uses `n_top=3` for file size, but the Animl handler convention (per `DEPLOYMENT_PROMPT.md`) is: "Returns a dict of `{class_name: confidence_score}` for ALL classes (no filtering)." The handler comment even says "Return ALL predictions" but then the `_results_to_images` in the reference only keeps top-3.

The correct output should be:
```python
scores = probabilities[0].tolist()
predictions = {classes[i]: scores[i] for i in range(len(scores))}
return predictions
```

---

## Windows path compat hack ⚠️ Unnecessary

The `pathlib.WindowsPath = pathlib.PosixPath` hack is copied from hwi-adsv1 but is not needed here. The hwi-adsv1 backbone is an `onnx2torch` GraphModule with potentially Windows-serialized paths. The small-animal-classifier checkpoint is a plain `torch.save` dict with no custom serialized path objects. It adds no harm, but it's dead code that will confuse future readers.

---

## Summary

| Area | Status | Issue |
|---|---|---|
| Model loading — checkpoint keys | ✅ Correct | — |
| Model loading — `weights_only` | ⚠️ Bug | Should be `False` (or try/fallback); `True` likely fails |
| Banner crop — logic | ✅ Correct | Correct copy of reference function |
| Banner crop — values | ⚠️ Minor | Hardcoded 0.03/0.035 instead of using checkpoint values |
| Resize interpolation | ⚠️ Bug | `transforms.Resize` ≠ PIL BILINEAR pre-tensor-conversion |
| Normalization | ✅ Correct | Values and order are right |
| Inference (forward pass, softmax) | ✅ Correct | — |
| Output format | ❌ Wrong | Should return `{class_name: score}` flat dict, not `{"images": [...]}` wrapper with index-keyed pairs |

The preprocessing resize and the output format are the two changes that would most affect correctness and integration with the rest of Animl.
