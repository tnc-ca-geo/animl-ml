# SpeciesNet

[SpeciesNet](https://github.com/google/cameratrapai/tree/main) is a collection machine learning models for species detection and classification in camera trap images from Google. The code here has components to containerize, test and deploy the model to AWS SageMaker to be used as part of ANIML.

## Components

1. **Dockerfiles**
There are two Dockerfiles:
  * `Dockerfile.cpu` for running the vanilla SpeciesNet LitServe server
  * `Dockerfile.sagemaker` for deploying to SageMaker.

2. **FastAPI Server**
The FastAPI Server (serve.py) provides support to deploy SpeciesNet on SageMaker. This wraps the SpeciesNet LitServe class and exposes the necessary /ping and /invocation API routes.

## Running LitServe Locally

1. **Build the Container**
   ```bash
   docker build -t speciesnet -f Dockerfile.cpu .
   ```

2. **Run the Server**
   ```bash
   docker run -p 8000:8000 speciesnet
   ```
   The server will be available at http://0.0.0.0:8000

3. **Example Request**
   ```bash
   curl --location 'http://0.0.0.0:8000/predict' \
   --header 'Content-Type: text/plain' \
   --data '{
       "instances": [
           {
               "filepath": "test_data/african_elephants.jpg"
           }
       ]
   }'
   ```

## Running SageMaker Container Locally

1. **Build the SageMaker Container**
   ```bash
   docker build -t speciesnet-sagemaker -f Dockerfile.sagemaker .
   ```

2. **Run the Container**
   ```bash
   docker run -p 8080:8080 speciesnet-sagemaker
   ```
   The server will be available at http://0.0.0.0:8080

3. **Test Endpoints**
   - Health check:
     ```bash
     curl http://localhost:8080/ping
     ```
   - Prediction endpoint: run the `tests/test_request.py` script: `cd tests && python test_request.py`

## Deploying to SageMaker

Use a SageMaker Notebook instance to run the `deploy_to_sagemaker.ipynb` notebook. The notebook walks through creating model on SageMaker, preparing the endpoint, deploying, and testing.