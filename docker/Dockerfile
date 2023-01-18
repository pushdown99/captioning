FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ENV CUDA_VERSION 11.1.1

LABEL maintainer "URBAN UNION <haeyun@gmail.com>"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y apt-utils 2>&1 | \
    grep -v "^debconf: delaying package configuration, since apt-utils.*"

RUN apt install build-essential -y --no-install-recommends

RUN apt-get install -yq python3 python3-pip vim wget git

RUN apt-get install -yq libgl1 libglib2.0-0

RUN apt-get update

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

RUN python -m pip install --upgrade pip

RUN pip install tensorflow colorama easydict
RUN pip install tqdm ipykernel nltk ipdb matplotlib pandas climage fire

ADD captioning /captioning

COPY motd /etc/motd
COPY getpubip /bin
COPY jupyter-run /root
COPY bashrc /root/.bashrc
COPY vimrc /root/.vimrc

RUN echo 'clear' >> /etc/bash.bashrc
RUN echo '[ ! -z "$TERM" -a -r /etc/motd ] && cat /etc/motd' >> /etc/bash.bashrc
RUN echo 'ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime' >> /etc/bash.bashrc

RUN python -m ipykernel install --user --name torch --display-name "torch"
RUN pip install gensim

WORKDIR /captioning

