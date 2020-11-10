#!/bin/bash
#
# Stop a local docker container.

set -euo pipefail

source ../sagemaker-tensorflow-serving-container/scripts/shared.sh

parse_std_args "$@"

docker kill $(docker ps -q --filter ancestor=tensorflow-inference:$full_version-$device)