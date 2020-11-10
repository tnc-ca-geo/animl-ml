#!/bin/bash
#
# Download megadetector model files

# Megadetector v4.1.0
md4Url=https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0_saved_model.zip
mdDir=megadetector
modelPath="$PWD/models"

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