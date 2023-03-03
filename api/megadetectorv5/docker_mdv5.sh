# $1 is the path to the dir holding the .mar file. there should be only one .mar file
docker run -p 8080:8080 -v $1:/opt/ml/model torchserve-mdv5a:0.5.3-cpu serve