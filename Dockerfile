FROM nvidia/cuda:10.0-base

ARG http_proxy
ENV http_proxy=$http_proxy
ENV https_proxy=$http_proxy

RUN apt-get update && \
    apt-get -y install build-essential && \
    apt-get -y install python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp
RUN pip3 --no-cache-dir install -r /tmp/requirements.txt

COPY . /usr/src/qurator-sbb-ner

RUN mkdir -p /usr/src/qurator-sbb-ner/konvens2019
RUN mkdir -p /usr/src/qurator-sbb-ner/digisam

RUN pip3 --no-cache-dir install -e /usr/src/qurator-sbb-ner

WORKDIR /usr/src/qurator-sbb-ner
CMD export LANG=C.UTF-8; env FLASK_APP=qurator/sbb_ner/webapp/app.py env FLASK_ENV=development env USE_CUDA=True flask run --host=0.0.0.0
