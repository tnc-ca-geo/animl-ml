FROM tensorflow/serving AS tfserving

FROM python:3.9-slim

# Install FastAPI and dependencies
COPY requirements.txt /opt/ml/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/requirements.txt

# Copy FastAPI app
COPY serve.py /opt/ml/code/serve.py

# Copy model from previous stage
COPY --from=tfserving /models /models
COPY --from=tfserving /usr/bin/tensorflow_model_server /usr/bin/tensorflow_model_server

# Set environment variables for model name
ENV MODEL_NAME=irc
ENV MODEL_VERSION=2

# Environment variables for SageMaker
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"
ENV MODEL_PATH=/opt/ml/models/${MODEL_NAME}/${MODEL_VERSION}/

# Copy our model folder to the container
COPY ./export/Servo/${MODEL_VERSION}/ ${MODEL_PATH}

# Copy class map to container
COPY ./index_to_name.json /opt/ml/code/

# Set working directory
WORKDIR /opt/ml/code

# Expose FastAPI and TF Serving ports
EXPOSE 8080 8501

# Start both servers
CMD ["bash", "-c", "uvicorn serve:app --host 0.0.0.0 --port 8080 & tensorflow_model_server --rest_api_port=8501 --model_name=irc --model_base_path=/opt/ml/models/irc"]