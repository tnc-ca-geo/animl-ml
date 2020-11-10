#!/bin/bash
#
# Download megadetector model files

# Megadetector v3
md3Url=https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/saved_model_megadetector_v3_tf19.zip
# Megadetector v4.1.0
md4Url=https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0_saved_model.zip
mdDir=megadetector
modelPath="$PWD/models"

if [ -d "$modelPath"/"$mdDir" ]; then
  echo -e "Directory "$modelPath"/"$mdDir" already exits, skipping ...\n"
else 
  echo -e "Downloading and unzipping megadetector v3..."
  cd "$modelPath" && \
  curl -sS "$md3Url" > md3.zip && \
  unzip md3.zip && rm md3.zip && \
  rm -r __MACOSX/ && \
  mv saved_model_megadetector_v3_tf19 "$mdDir"
  cd $mdDir
  mv 1 3
  cd 3
  pbFile=$(find . -name '*.pb')
  echo -e "renaming .pb file: $pbFile..."
  mv "$pbFile" "saved_model.pb"
  cd ..
  echo -e "Downloading and unzipping megadetector v4..."
  curl -sS "$md4Url" > md4.zip && \
  unzip md4.zip && rm md4.zip && \
  mv saved_model 4
  cd ..
  find . -name '*.DS_Store' -type f -delete
  echo -e "Success"
fi