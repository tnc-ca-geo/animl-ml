# $1 is the path to the dir holding the .mar file. there should be only one .mar file
echo $1
docker run -it -p 8080:8080 -v $1:/opt/ml/model torchserve-deepfaune-ne:latest-cpu serve