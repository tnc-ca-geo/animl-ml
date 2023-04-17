# $1 is the path to the dir holding the .mar file. there should be only one .mar file
docker run -it -p 8080:8080 -v $1:/opt/ml/model torchserve-nzdoc:latest-cpu serve