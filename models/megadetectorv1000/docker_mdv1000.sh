#!/bin/bash
# $1 is the path to the dir holding the .mar file. there should be only one .mar file
docker run -it -p 8080:8080 -p 8081:8081 -p 8082:8082 -v $1:/opt/ml/model torchserve-mdv1000:0.11.0-cpu serve
