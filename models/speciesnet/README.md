# SpeciesNet

[SpeciesNet](https://github.com/google/cameratrapai/tree/main) is a collection machine learning models for species detection and classification in camera trap images from Google. The code here has components to containerize, test and deploy the model to AWS SageMaker to be used as part of Animl.

## Components

1. **Dockerfiles**
   There are two Dockerfiles:

- `Dockerfile.cpu` for running the vanilla SpeciesNet LitServe server
- `Dockerfile.sagemaker` for deploying to SageMaker.

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
   - Run automated tests:

     ```bash
     cd tests && python -m pytest test_request.py
     ```

     The tests verify:

     - Health check endpoint functionality
     - Default settings (both detection and classification)
     - Classification-only mode
     - Detection-only mode
     - Without geofencing

     Each test validates the response structure and the number of predictions (detections/classifications) returned by the model.

## Deploying to SageMaker

Use a SageMaker Notebook instance to run the `deploy_to_sagemaker.ipynb` notebook. The notebook walks through creating model on SageMaker, preparing the endpoint, deploying, and testing.

## Misc

**Taxonomy Transformer**
The `transform_taxonomy.py` script converts semicolon-separated taxonomy data into JSON format. The label data is available when [you download the model](https://www.kaggle.com/models/google/speciesnet/pyTorch). It creates a JSON file which is loaded to the animl-api MongoDB. To run: `python transform_taxonomy.py input_file output_file`
