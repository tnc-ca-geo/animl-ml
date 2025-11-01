python yolov5/export.py --imgsz '(960,1280)' --weights model-weights/md_v5a.0.0.pt --include torchscript onnx
mv model-weights/md_v5a.0.0.onnx model-weights/md_v5a.0.0.960.1280.onnx
mv model-weights/md_v5a.0.0.torchscript model-weights/md_v5a.0.0.960.1280.torchscript
python yolov5/export.py --imgsz '(1280,1280)' --weights model-weights/md_v5a.0.0.pt --include torchscript onnx
mv model-weights/md_v5a.0.0.onnx model-weights/md_v5a.0.0.1280.1280.onnx
mv model-weights/md_v5a.0.0.torchscript model-weights/md_v5a.0.0.1280.1280.torchscript
python yolov5/export.py --imgsz '(642,856)' --weights model-weights/md_v5a.0.0.pt --include torchscript onnx
mv model-weights/md_v5a.0.0.onnx model-weights/md_v5a.0.0.642.856.onnx
mv model-weights/md_v5a.0.0.torchscript model-weights/md_v5a.0.0.642.856.torchscript
python yolov5/export.py --imgsz '(642,642)' --weights model-weights/md_v5a.0.0.pt --include torchscript onnx
mv model-weights/md_v5a.0.0.onnx model-weights/md_v5a.0.0.642.642.onnx
mv model-weights/md_v5a.0.0.torchscript model-weights/md_v5a.0.0.642.642.torchscript