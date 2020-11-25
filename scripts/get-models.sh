#!/bin/bash

# Download megadetector model files
# After running get-models.sh, the models/ directory should look like:
#
# models
#   ├─ megadetector
#       └─ megadetector
#           └─ 4
#               └─ saved_model.pb
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

bucket="s3://animl-models"
modelPath="$PWD/models"

# Microsoft Megadetector v4
megadetectorDir="megadetector"
megadetectorModel="megadetector"
megadetectorZipFile="megadetector.zip"
megadetectorPbPath="$modelPath/$megadetectorDir/megadetector/4/saved_model.pb"

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

if [[ -f $megadetectorPbPath ]]; then
  echo -e "$megadetectorModel model file already exits, skipping"
else 
  echo -e "Downloading and unzipping Megadetector v4..."
  cd $modelPath/$megadetectorDir
  aws s3 cp $bucket/$megadetectorZipFile ./ && \
  unzip $megadetectorZipFile && rm $megadetectorZipFile && \
  cd ..
  find . -name "*.DS_Store" -type f -delete
  echo -e "Success"
fi 

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
