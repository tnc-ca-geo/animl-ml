FROM pytorch/torchserve:0.5.3-cpu
RUN whoami
RUN ls -la /home/venv/bin/pip
USER root
# RUN pip install --upgrade pip && pip install opencv-python ipython
# commit id https://github.com/ultralytics/yolov5/blob/9286336cb49d577873b2113739788bbe3b90f83c/requirements.txt
RUN pip install gitpython ipython matplotlib>=3.2.2 numpy==1.23.4 opencv-python==4.6.0.66 \
    Pillow==9.2.0  psutil PyYAML>=5.3.1 requests>=2.23.0 scipy==1.9.3 thop>=0.1.1 \
    torch==1.10.0 torchvision==0.11.1 tqdm>=4.64.0 tensorboard>=2.4.1 pandas>=1.1.4 \
    seaborn>=0.11.0 setuptools>=65.5.1    
COPY ./deployment/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
RUN mkdir -p /home/model-server/ && mkdir -p /home/model-server/tmp
COPY ./deployment/config.properties /home/model-server/config.properties
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
# ENV YOLOv5_AUTOINSTALL=False
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]