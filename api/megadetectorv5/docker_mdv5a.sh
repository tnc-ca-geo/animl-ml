# $1 is the path to the dir holding the .mar file. there should be only one .mar file
docker run -it -p 8080:8080 -p 8081:8081 -p 8082:8082 -v $1:/opt/ml/model torchserve-mdv5a:0.5.3-cpu serve