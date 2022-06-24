# Setup Instructions

## Download weights and torchscript model

```
cd animl-ml/api/megadetectorv5
aws s3 sync s3://animl-model-zoo/megadetectorv5/ models/megadetectorv5/
```

## Export yolov5 weights as torchscript model
Size needs to be same as in mdv5_handler.py for good performance 
```
python yolov5/export.py --weights models/megadetectorv5/md_v5a.0.0.pt --img 640 640 --batch 1 
```
this will create models/megadetectorv5/md_v5a.0.0.torchscript 

## Run model archiver

```
torch-model-archiver --model-name mdv5 --version 1.0.0 --serialized-file models/megadetectorv5/md_v5a.0.0.torchscript --extra-files index_to_name.json --handler mdv5_handler.py
mkdir -p model_store
mv mdv5.mar model_store/megadetectorv5-yolov5-1-batch-640-640.mar
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
