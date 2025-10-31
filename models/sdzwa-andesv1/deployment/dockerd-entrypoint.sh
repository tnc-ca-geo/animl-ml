#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    printenv
    ls /opt
    ls /opt/ml
    ls /opt/ml/model
    torchserve --start --ts-config /home/model-server/config.properties
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null