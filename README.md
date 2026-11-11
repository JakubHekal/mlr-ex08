# mlr-ex08 — Leaf Classifier

Live web page: **https://jakubhekal.github.io/mlr-ex08/**

Classifies leaves (apple, cherry, chestnut, maple) live from the phone's rear camera using `leaf_model.onnx` with [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/). A photo upload is provided as a fallback.

Preprocessing follows `model_info.json` exactly: central square crop, resize to 96×96, RGB scaled to 0–1, per-channel standardization with ImageNet mean/std, input tensor `1×3×96×96`.

The QR code at the bottom of the page opens the same public GitHub Pages URL and was generated locally (no external QR-image service).
