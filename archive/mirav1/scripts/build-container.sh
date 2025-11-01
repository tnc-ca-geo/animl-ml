#!/bin/bash
#
# Build the docker image
# NOTE: hardcoding image pull & build for now due issues with 
# sagemaker-tensorflow-serving-container's build scripts:
# https://github.com/aws/sagemaker-tensorflow-serving-container/issues/175

# TensorFlow 1.12 - not working
# aws_account=520713654638
# repository=sagemaker-tensorflow-serving
# full_version=1.12.0
# short_version=1.12
# device=cpu
# aws_region=us-west-1

# TensorFlow 1.13
aws_account=763104351884
repository=tensorflow-inference
full_version=1.13.0
short_version=1.13
device=cpu
aws_region=us-west-1

set -euo pipefail

echo "pulling previous image for layer cache... "
aws ecr get-login-password --region ${aws_region} \
    | docker login \
        --password-stdin \
        --username AWS \
        "${aws_account}.dkr.ecr.${aws_region}.amazonaws.com/${repository}" || echo 'warning: ecr login failed'
docker pull $aws_account.dkr.ecr.$aws_region.amazonaws.com/$repository:$full_version-$device || echo 'warning: pull failed'
docker logout https://$aws_account.dkr.ecr.$aws_region.amazonaws.com

echo "building image... "
cp -r ../sagemaker-tensorflow-serving-container/docker/build_artifacts/* ../sagemaker-tensorflow-serving-container/docker/$short_version/
docker build \
    --cache-from $aws_account.dkr.ecr.$aws_region.amazonaws.com/$repository:$full_version-$device \
    --build-arg TFS_VERSION=$full_version \
    --build-arg TFS_SHORT_VERSION=$short_version \
    -f ../sagemaker-tensorflow-serving-container/docker/$short_version/Dockerfile.$device \
    -t $repository:$full_version-$device \
    -t $repository:$short_version-$device \
    ../sagemaker-tensorflow-serving-container/docker/$short_version/
