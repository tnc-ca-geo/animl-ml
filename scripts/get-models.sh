#!/bin/bash
#
# Download megadetector model files

# TODO: store zipped up MIRA models in TensorFlow ProtoBuf (.pb) format, 
# and download them & create any necessary folder structure.
# After running get-models.sh, the models/ directory should look like: 

# models
#   ├─ megadetector
#       └─ megadetector
#           └─ 4
#               └─ saved_model.pb
#   ├─ mira
#       └─ mira-large
#           └─ 1
#               └─ saved_model.pb
#                   └─ variables
#                       └─ variables.data-00000-of-00001
#                       └─ variables.index
#       └─ mira-small
#           └─ 1
#               └─ saved_model.pb
#                   └─ variables
#                       └─ variables.data-00000-of-00001
#                       └─ variables.index
#       └─ code
#           └─ inference.py
#           └─ requirements.txt


# Megadetector v4.1.0
md4Url=https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0_saved_model.zip
mdDir=megadetector
modelPath="$PWD/models/megadetector"

if [ -d "$modelPath"/"$mdDir" ]; then
  echo -e "Directory "$modelPath"/"$mdDir" already exits, skipping ...\n"
else 
  echo -e "Downloading and unzipping megadetector v4..."
  cd "$modelPath" && \
  mkdir "$mdDir" && cd $mdDir
  curl -sS "$md4Url" > md4.zip && \
  unzip md4.zip && rm md4.zip && \
  mv saved_model 4
  cd ..
  find . -name '*.DS_Store' -type f -delete
  echo -e "Success"
fi