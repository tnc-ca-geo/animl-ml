FROM pytorch/torchserve:0.5.3-cpu
RUN whoami
RUN ls -la /home/venv/bin/pip
USER root
# RUN pip install --upgrade pip && pip install opencv-python ipython
# commit id https://github.com/ultralytics/yolov5/blob/9286336cb49d577873b2113739788bbe3b90f83c/requirements.txt
# handling libjpeg version by installing from source
RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        mercurial \
        openssh-client \
        procps \
        subversion \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV CONDA_DIR /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
# Leave these args here to better use the Docker build cache
ARG CONDA_VERSION=py38_22.11.1-1
RUN set -x && \
    UNAME_M="$(uname -m)" && \
    if [ "${UNAME_M}" = "x86_64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh"; \
        SHA256SUM="00938c3534750a0e4069499baf8f4e6dc1c2e471c86a59caa0dd03f4a9269db6"; \
    elif [ "${UNAME_M}" = "s390x" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-s390x.sh"; \
        SHA256SUM="a150511e7fd19d07b770f278fb5dd2df4bc24a8f55f06d6274774f209a36c766"; \
    elif [ "${UNAME_M}" = "aarch64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-aarch64.sh"; \
        SHA256SUM="48a96df9ff56f7421b6dd7f9f71d548023847ba918c3826059918c08326c2017"; \
    elif [ "${UNAME_M}" = "ppc64le" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-ppc64le.sh"; \
        SHA256SUM="4c86c3383bb27b44f7059336c3a46c34922df42824577b93eadecefbf7423836"; \
    fi && \
    wget "${MINICONDA_URL}" -O miniconda.sh -q
RUN mkdir -p /opt && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy
RUN conda update conda
RUN conda install -c pytorch -c powerai -c conda-forge torchserve gitpython ipython matplotlib>=3.2.2 numpy==1.23.4 \
    Pillow==9.2.0 psutil PyYAML>=5.3.1 requests>=2.23.0 scipy==1.9.3 \
    tqdm>=4.64.0 tensorboard>=2.4.1 pandas>=1.1.4 \
    seaborn>=0.11.0 pytorch==1.10.0 opencv==4.6 torchvision==0.11.1
RUN pip install thop>=0.1.1 
USER model-server