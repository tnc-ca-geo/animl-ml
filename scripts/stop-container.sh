#!/bin/bash
#
# Stop a local docker container.

set -euo pipefail

full_version=1.13.0
device=cpu

docker kill $(docker ps -q --filter ancestor=tensorflow-inference:$full_version-$device)