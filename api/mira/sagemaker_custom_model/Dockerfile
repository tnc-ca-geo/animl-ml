FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:1.14-cpu
COPY requirements.txt /home/requirements.txt
RUN pip install -r /home/requirements.txt
# ENV PATH="/opt/ml/code:${PATH}"
# # COPY ../models/mira-large/code /opt/ml/code
# # WORKDIR /opt/ml/code
# ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
# ENV SAGEMAKER_PROGRAM inference.py
