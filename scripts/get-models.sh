#!/bin/bash
#
# Download megadetector model files

# Megadetector v3
mdURL=https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/saved_model_megadetector_v3_tf19.zip
originalPBFile=megadetector_v3_tf19_saved_model.pb

# # Megadetector v4.1.0
# mdURL=https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0_saved_model.zip
# originalPbFile=md_v4.1.0_saved_model.pb

mdDir=megadetector
modelPath="$PWD/models"

if [ -d "$modelPath"/"$mdDir" ]; then
  echo -e "Directory "$modelPath"/"$mdDir" already exits, skipping ...\n"
else 
  echo -e "Downloading and unzipping megadetector model..."
  cd "$modelPath" && \
  curl -sS "$mdURL" > md.zip && \
  unzip md.zip && rm md.zip && \
  rm -r __MACOSX/ && \
  pbFile=$(find . -name '*.pb')
  pbDir=$(dirname "$pbFile")
  echo -e "renaming .pb file: $pbFile..."
  mv "$pbFile" "$pbDir/saved_model.pb"
  find . -name '*.DS_Store' -type f -delete
fi