# Animl ML
Machine Learning resources for camera trap data processing

## Intro

We are using AWS Sagemaker to host our model endpoints. The initial models
we will run inference on are 
[Microsoft's Megadetector](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md),
an model to detect animals, people, and vehicles in camera trap images, and 
[MIRA](https://github.com/tnc-ca-geo/mira), a species classifier trained on 
labeled images from Santa Cruz Island.

When you deploy model endpoints to Sagemaker, AWS starts an EC2 instance and 
starts a docker container with in it optimized for serving particular model's 
architecture. You can find Phython Nodetbooks to facilitate the deployment of 
models in the notebooks/ directory of this repo.

Additionally, if you want to launch a TensorFlow serving container locally 
to debug and test endpoints locally before deploying, this repo contains git 
submodule verisons of AWS's 
[Sagemaker TensorFlow Serving Container repo](https://github.com/aws/sagemaker-tensorflow-serving-container/) 
and [Microsoft's CameraTraps repo](https://github.com/microsoft/CameraTraps) 
and instructions below to help you run the container, load models, dependencies, 
and pre/postprocessing scripts into it, and submit requests to the local 
endpoints for inference.

## Getting started

### Cloning the repo
BEFORE YOU CLONE: Because this repo contains 
[git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules), you 
need to clone recursively in order to pull down the files from the other repos:

```
$ git clone --recursive https://github.com/tnc-ca-geo/animl-ml/
```

If you forget to clone recursively, just run the following to retrieve the 
submodules:

```
$ git submodule init
$ git submodule update
```

### Pulling upstream changes from this repo
If you simply run ```git pull``, it will recursively fetche submodules changes, 
however, it does not ***update*** the submodules. To update them you'll need to 
run:

```
$ git pull
$ git submodule update --init --recursive
```

### Fetching upstream submodule updates
To pull updates from the remote repos tracked in our submodules, run: 
```
$ git submodule update --remote [submodule]
```
The ```[submodule]``` part is optional. If you leave that out git will fetch 
and merge updates from all submodules.

### Hacking on the submodules
The reason we're using git submodules to pull in these repos is because we need 
to make some small changes in order to get them running for our purposes. 
However, as the git docs explain: 

>"when we’ve run the ```git submodule update``` command to fetch changes from the 
>submodule repositories, Git would get the changes and update the files in the 
>subdirectory but will leave the sub-repository in what’s called a “detached 
>HEAD” state. This means that there is no local working branch (like master, 
>for example) tracking changes. With no working branch tracking changes, that 
>means even if you commit changes to the submodule, those changes will quite 
>possibly be lost the next time you run git submodule update. You have to do 
>some extra steps if you want changes in a submodule to be tracked.
>
>In order to set up your submodule to be easier to go in and hack on, you need 
>to do two things. You need to go into each submodule and check out a branch to 
>work on. Then you need to tell Git what to do if you have made changes and 
>then git ```submodule update --remote``` pulls in new work from upstream. The 
>options are that you can merge them into your local work, or you can try to 
>rebase your local work on top of the new changes."

## Local development and experimentation

NOTE: this assumes that you have 
[aws-vault](https://github.com/99designs/aws-vault) installed. 



