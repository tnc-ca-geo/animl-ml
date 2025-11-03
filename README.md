# Animl ML

This repo contains code and documentation for deploying ML models for camera trap image processing to [Animl](https://animl.camera), as well as code and documentation for training custom classifiers.

## Model hosting and deployment

### Overview

Animl uses [AWS Sagemaker Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html) endpoints for model hosting and inference. Serverless Inference endpoints are appealing for intermittent, spiky loads because they auto-scale to zero when not in use. This is particularly useful for hosting n-number of localized models, some of which are used very infrequently, which is the case for Animl.

SageMaker Serverless endpoints are also CPU-based, rather than GPU, which offers cost savings but means that (a) models often need to be compiled before deployment to run on CPUs and (b) inference is slower. To compensate for the slower inference, when requests come in we scale out horizontally, spinning up many Serverless Inference endpoints at once to process the images in parallel (more specifics on how the concurrent processing is configured can be found [here](https://github.com/tnc-ca-geo/animl-api/issues/101)).

### Deploying a new model

If you're interested in deploying a model to Animl, start by gathering the following information from the model developer:

1.  Model weights, class list/mappings files, and any available metadata about the model
2.  Which deep learning library was the model trained in (e.g. PyTorch, TensorFlow), and what version?
3.  What model architecture was used (e.g. ResNet, EfficientNet), and what version/size?
4.  What size inputs does the model expect (e.g. 299x299)
5.  Does the developer have training and/or inference scripts available to reference? Are there any preprocessing steps (such as image augmentations/transformations) or postprocessing steps (such as heuristic decision making) to be aware of? We may need to re-implement them in our handler functions.
6.  Does the developer have a recommended default confidence threshold?

The deployment process will vary depending on whether the model was trained in PyTorch or TensorFlow, but the workflow for each broadly follows the same pattern. We recommend copying existing deployment code from the `/models` directory and adapting it for your needs. More specifically:

- for **PyTorch** models, we recommend referencing [deepfaune-ne](/models/deepfaune-ne/), [sdzwa-southwestv3](/models/sdzwa-southwestv3/), or [megadetectorv5](/models/megadetectorv5/) as starting points.
- for **TensorFlow** models, we recommend referencing [irc](/models/irc) as a starting point.
- for **ensemble** models, we recommend referencing [speciesnet](/models/speciesnet/) as a starting point.

For all models, however, the deployment workflow follows these steps:

#### Step 1: Prepare model, handler functions and test container locally

We use the "bring your own container" [approach](https://sagemaker-examples.readthedocs.io/en/latest/advanced_functionality/scikit_bring_your_own/scikit_bring_your_own.html) for all of our models deployed to SageMaker, which gives us a lot of control over model invocation and allows us to build and test our containers locally. The exact stack of the container will vary depending on training framework (for PyTorch, we start from Torchserve Docker images; for TensorFlow, we use TensorFlow Serving wrapped in a FastAPI app). You also may need to load the model into memory on your local machine and compile it to a different model format (Torchscript for PyTorch, Saved Model bundle for Tensorflow), and adjust the handler functions to meet the expectations of the new model.

Because the implementation details vary, as mentioned above we recommend starting from existing examples in the `/models` directory, but broadly speaking you'll likely have to perform the following steps:

1. convert or compile the model in some way to make it suitable for serving
2. adjust pre/post processing handler functions as needed
3. build and test your container locally
4. upload the compiled model for use in the next step

#### Step 2: Deploy model to Sagemaker Serverless Inference endpoint using Sagemaker Notebook

Each model in the `/models` directory has a `deploy_to_sagemaker.ipynb`, which are intended to be run in hosted Sagemaker Notebooks. To deploy your model:

1. copy a deployment notebook from an existing endpoint
2. create a Sagemaker Notebook instance and associate it with https://github.com/tnc-ca-geo/animl-ml repo
3. step through the deployment notebook, adjusting as needed, and test deployed endpoint(s)
4. DON’T FORGET TO DEACTIVATE NOTEBOOK WHEN FINISHED!

#### Step 3: Add SSM params

The [animl-api](https://github.com/tnc-ca-geo/animl-api) uses SSM parameters to map the deployed model endpoint names to config variables referenced at runtime. You will need to create 2 new parameters (one for `dev` and one for `prod`) for each of the endpoints you deployed. So if you deployed one endpoint for batch inference and one for real-time inference, you'll need to create a total of 4 new parameters following the naming convention below:

| Parameter Name                            | Parameter Value                                   |
| ----------------------------------------- | ------------------------------------------------- |
| `/ml/<model_name>-batch-endpoint-dev`     | `<model_name>-concurrency-<batch_concurrency>`    |
| `/ml/<model_name>-batch-endpoint-prod`    | `<model_name>-concurrency-<batch_concurrency>`    |
| `/ml/<model_name>-realtime-endpoint-dev`  | `<model_name>-concurrency-<realtime_concurrency>` |
| `/ml/<model_name>-realtime-endpoint-prod` | `<model_name>-concurrency-<realtime_concurrency>` |

#### Step 4: In animl-api, fetch new SSM params and implement model interface

See [this PR](https://github.com/tnc-ca-geo/animl-api/pull/368/files) as an example.

#### Step 5: Add `MLModel` record to MongoDB

You'll need to create an `MLModel` record with an array of `categories` for each class your model predicts. For models with a small number of classes it's easy to clone and amend an existing record in the DB, for models with a large number of classes we recommend scripting the category creation (see SpeciesNet example of that here [here](/models/speciesnet/transform_taxonomy.py)).

> [!NOTE]  
> Don’t forget to scrub periods (`.`) and dollar signs (`$`) from [MLModel.categories.name](http://MLModel.categories.name) fields when creating your `MLModel` records. More info on why we need to do so can be found [here](https://github.com/tnc-ca-geo/animl-api/issues/314).

Lastly, to make the model available to Animl Projects for use in their [Automation Rules](https://docs.animl.camera/fundamentals/automation-rules), you'll need to add the new `MLModel._id` to each Projects' `Project.availableMLModels`.

### Other things to note

- **Real-time vs. batch endpoints:** Animl can ingest camera trap images in two ways: integrated wireless/cellular cameras can send their images to Animl for processing in real-time, or users can bulk upload a zip file of images directly from their local computers. In order to prevent large bulk/batch uploads from blocking real-time inference requests, for many of the models we host we maintain separate endpoints for each ingestion pathway: one for processing large batch uploads and one for processing smaller numbers of wireless images in real-time. For more information on how the two sets of endpoints are integrated in the larger application, high-level documentation and a diagram of the Animl architecture can be found [here](https://github.com/tnc-ca-geo/animl-api/tree/main/documentation). For the purposes of this repo, it's worth being aware that when we need to support real-time inference for a model, we deploy them as separate endpoints with different concurrency settings. More on that below.
- **AWS concurrency limit:** As mentioned above we compensate for Serverless Endpoint latency (both from cold-starts and slower inference on CPUs) by scaling horizontally and making maximal use of our AWS account's endpoint concurrency allotment (1000 per region). As of November 2025 we’re already using 940 out of our 1000 allotted concurrencies in the US-West-2 region. If we want to host more, my understanding is that our options are:
  a. dial down the concurrency settings of our current models to make room for more endpoints in the region
  b. standing up Animl on multiple AWS accounts. E.g. maintain one for TNC users, one for external partners, an separate ones for enterprise users.
  c. ask AWS to increase our quota
- **Animl prod & dev stacks use the same endpoints:** The only reason for this is to conserve our concurrency quota.
- **A note about cost and scalability:** Initially Animl used models deployed on Sagemaker Realtime Endpoints, which were expensive because we were paying for an on-demand GPU instance even when not in use. Switching to Serverless Endpoints reduced our inference costs by roughly 10x at the time. However, because Serverless Inference endpoints are pay-per-use, at some point (I suppose at >10x our current usage), the fixed-priced, dedicated endpoints we had been using previously will begin to be the more cost-effective option. Another likley cheaper approach would be to hold the inference requests in a queue and spin up/tear down processing resources on a schedule.

## Classifier training

Instructions for training custom classifiers can be found in the [/classifier-training](/classifier-training/) directory.

## Related repos

Animl is comprised of a number of microservices, most of which are managed in their own repositories.

### Core services

Services necessary to run Animl:

- [Animl Ingest](http://github.com/tnc-ca-geo/animl-ingest)
- [Animl API](http://github.com/tnc-ca-geo/animl-api)
- [Animl Frontend](http://github.com/tnc-ca-geo/animl-frontend)
- [EXIF API](https://github.com/tnc-ca-geo/exif-api)

### Wireless camera services

Services related to ingesting and processing wireless camera trap data:

- [Animl Base](http://github.com/tnc-ca-geo/animl-base)
- [Animl Email Relay](https://github.com/tnc-ca-geo/animl-email-relay)
- [Animl Ingest API](https://github.com/tnc-ca-geo/animl-ingest-api)

### Misc. services

- [Animl ML](http://github.com/tnc-ca-geo/animl-ml)
- [Animl Analytics](http://github.com/tnc-ca-geo/animl-analytics)
