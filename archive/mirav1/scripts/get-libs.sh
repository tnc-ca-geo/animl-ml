#!/bin/bash
#
# Clone sagmaker-tensorflow-serving-container 
# and Microsoft's CameraTrap repos

parentPath=$(cd ../ && pwd)
cameraTrapDir=CameraTraps
cameraTrapRepo=https://github.com/microsoft/CameraTraps
cameraTrapCommit=625c1f430bc0c832951a9707a97d446b57e3741e
sagemakerContainerDir=sagemaker-tensorflow-serving-container
sagemakerContainerRepo=https://github.com/aws/sagemaker-tensorflow-serving-container
sagemakerContainerCommit=fc9013c5cc6cb521585b89aa2c984cc93864f445

# get cameratrap repo
if [ -d "$parentPath"/"$cameraTrapDir" ]; then
  echo -e "Directory "$parentPath"/"$cameraTrapDir" already exits, pulling down latest commits ...\n"
  pullCmdRun=$(
    cd "$parentPath/$cameraTrapDir" 
    git checkout master 
    git pull
  )
  echo -e "${pullCmdRun}\n\n"
else
  echo -e "No dir found for "$parentPath"/"$cameraTrapDir", cloning remote ...\n"
  cloneCmd="git clone $cameraTrapRepo $parentPath/$cameraTrapDir"
  cloneCmdRun=$(
    $cloneCmd
    # uncomment below to create a branch at specific commit
    # $cloneCmd &&
    # cd "$parentPath/$cameraTrapDir"
    # git checkout "$cameraTrapCommit"
    # git checkout -b pinned-commit
  )
  echo -e "${cloneCmdRun}\n\n"
fi

# get sagemaker container repo
if [ -d "$parentPath"/"$sagemakerContainerDir" ]; then
  echo -e "Directory "$parentPath"/"$sagemakerContainerDir" already exits, pulling down latest commits ...\n"
    pullCmdRun=$(
      cd "$parentPath/$sagemakerContainerDir" 
      git checkout master 
      git pull
    )
  echo -e "${pullCmdRun}\n\n"
else
  echo -e "No dir found for "$parentPath"/"$sagemakerContainerDir", cloning remote ...\n"
  cloneCmd="git clone $sagemakerContainerRepo $parentPath/$sagemakerContainerDir"
  cloneCmdRun=$(
    $cloneCmd
    # uncomment below to create a branch at specific commit
    # $cloneCmd &&
    # cd "$parentPath/$sagemakerContainerDir"
    # git checkout "$sagemakerContainerCommit"
    # git checkout -b pinned-commit
  )
  echo -e "${cloneCmdRun}\n\n"
fi
