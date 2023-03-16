FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ENV PYTHONUNBUFFERED TRUE

# FASTAI
RUN apt-get update && apt-get install -y software-properties-common rsync
RUN add-apt-repository -y ppa:git-core/ppa && apt-get update && apt-get install -y git libglib2.0-dev && apt-get update
# TORCHSERVE
RUN DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    fakeroot \
    ca-certificates \
    dpkg-dev \
    g++ \
    openjdk-11-jdk \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# TORCHSERVE
RUN git clone https://github.com/pytorch/serve.git
RUN cd serve && git reset --hard 490fbdb9dd4d13b3d914cb47dcc4daa2c02fedf5
RUN pip install ./serve/
RUN pip install captum

COPY ./deployment/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh

RUN mkdir -p /home/model-server/ && mkdir -p /home/model-server/tmp
COPY ./deployment/config.properties /home/model-server/config.properties

WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]
