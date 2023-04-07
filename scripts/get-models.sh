#!/bin/bash

# Download mira model files
# After running get-models.sh, the models/ directory should look like:
#
# models
#   ├─ mira-large
#       └─ code
#           └─ inference.py
#           └─ requirements.txt
#       └─ mira-large
#           └─ 1
#               └─ saved_model.pb
#               └─ variables
#                   └─ variables.data-00000-of-00001
#                   └─ variables.index
#   ├─ mira-small
#       └─ code
#           └─ inference.py
#           └─ requirements.txt
#       └─ mira-small
#           └─ 1
#               └─ saved_model.pb
#               └─ variables
#                   └─ variables.data-00000-of-00001
#                   └─ variables.index

bucket="s3://animl-model-zoo"
modelPath="$PWD/models"

# Mira large
miraLargeDir="mira-large"
miraLargeModel="mira-large"
miraLargeZipFile="mira-large.zip"
miraLargePbPath="$modelPath/$miraLargeDir/mira-large/1/saved_model.pb"

# Mira small
miraSmallDir="mira-small"
miraSmallModel="mira-small"
miraSmallZipFile="mira-small.zip"
miraSmallPbPath="$modelPath/$miraSmallDir/mira-small/1/saved_model.pb"

echo -e "Getting Animl models..."

if [[ -f $miraLargePbPath ]]; then
  echo -e "$miraLargeModel model file already exits, skipping"
else   echo -e "Downloading and unzipping MIRA-large..."
  cd $modelPath/$miraLargeDir
  aws s3 cp $bucket/$miraLargeZipFile ./ && \
  unzip $miraLargeZipFile && rm $miraLargeZipFile && \
  cd ..
  find . -name "*.DS_Store" -type f -delete
  echo -e "Success"
fi

if [[ -f $miraSmallPbPath ]]; then
  echo -e "$miraSmallModel model file already exits, skipping"
else   echo -e "Downloading and unzipping MIRA-small..."
  cd $modelPath/$miraSmallDir
  aws s3 cp $bucket/$miraSmallZipFile ./ && \
  unzip $miraSmallZipFile && rm $miraSmallZipFile && \
  cd ..
  find . -name "*.DS_Store" -type f -delete
  echo -e "Success"
fi

echo -e "Done"
