# Setup Instructions

## Download weights and torchscript model
From this directory, run:
```
aws s3 sync s3://animl-model-zoo// models/
```

## Export yolov5 weights as torchscript model

first, clone and install yolov5 dependencies and yolov5 following these instructions: https://docs.ultralytics.com/tutorials/torchscript-onnx-coreml-export/

Size needs to be same as in mdv5_handler.py for good performance. Run this from this directory 
```
python ../../../yolov5/export.py --weights models/md_v5a.0.0.pt --img 1280 1280 --batch 1 
```
this will create models/md_v5a.0.0.torchscript 

## Run model archiver
first, `pip install torch-model-archiver` then,

```
torch-model-archiver --model-name mdv5 --version 1.0.0 --serialized-file models//md_v5a.0.0.torchscript --extra-files index_to_name.json --handler mdv5_handler.py
mkdir -p model_store
mv mdv5.mar model_store/megadetectorv5-yolov5-1-batch-1280-1280.mar
```

The .mar file is what is served by torchserve.

## Serve the torchscript model with torchserve

```
bash docker_mdv5.sh
```

## Return prediction in normalized coordinates with category integer and confidence score

```
curl http://127.0.0.1:8080/predictions/mdv5 -T ../../input/sample-img-fox.jpg
```
