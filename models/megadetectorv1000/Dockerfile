FROM pytorch/torchserve:0.11.0-cpu
USER root
RUN pip install megadetector
COPY ./deployment/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
RUN mkdir -p /home/model-server/ && mkdir -p /home/model-server/tmp
COPY ./deployment/config.properties /home/model-server/config.properties
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]
