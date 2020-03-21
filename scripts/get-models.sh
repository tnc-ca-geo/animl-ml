#!/bin/bash
#
# Clone sagmaker-tensorflow-serving-container 
# and Microsoft's CameraTrap repos

mdURL=https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/saved_model_megadetector_v3_tf19.zip
originalPBFile=megadetector_v3_tf19_saved_model.pb
mdDir=megadetector
modelPath="$PWD/models"

# get cameratrap repo
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
fi