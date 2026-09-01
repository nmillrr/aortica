# Bundled edge model

`model.onnx` is the INT8 edge model loaded by `OnnxInferenceEngine`
(`MODEL_FILENAME = "model.onnx"`). It is committed here so a fresh clone
builds an APK that works offline — which is the point of the Android client.

Do not edit it by hand. It is a copy of the current release artifact in
`artifacts_combined/`, kept in sync by:

```
python3 scripts/sync_release_artifacts.py
```

CI runs that script with `--check` and fails if this file is missing, stale,
or under 100 KB. That floor exists because this directory previously held no
model at all, and the PWA's equivalent held a two-byte placeholder that ONNX
Runtime accepted as a valid protobuf prefix.

`model_metadata.json` is optional and is written by the export pipeline:

```python
from aortica.edge.mobile_export import export_mobile_model
export_mobile_model("aortica_edge_int8.onnx", "mobile/android/app/src/main/assets/")
```
