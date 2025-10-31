FROM pytorch/torchserve:latest-cpu
RUN whoami
RUN ls -la /home/venv/bin/pip
USER root
# RUN pip install --upgrade pip && pip install opencv-python ipython
RUN pip install gitpython ipython matplotlib>=3.2.2 numpy==1.23.4 opencv-python==4.6.0.66 \
    Pillow==9.2.0 psutil PyYAML>=5.3.1 requests>=2.23.0 scipy==1.9.3 thop>=0.1.1 \
    tqdm>=4.64.0 tensorboard>=2.4.1 pandas>=1.1.4 \
    seaborn>=0.11.0 setuptools>=65.5.1
RUN  pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
COPY ./deployment/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
RUN mkdir -p /home/model-server/ && mkdir -p /home/model-server/tmp
COPY ./deployment/config.properties /home/model-server/config.properties
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
# ENV YOLOv5_AUTOINSTALL=False
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]