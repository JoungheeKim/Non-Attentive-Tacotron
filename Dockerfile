FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install some basic utilities - required
RUN . /etc/os-release; \
                printf "deb http://ppa.launchpad.net/jonathonf/vim/ubuntu %s main" "$UBUNTU_CODENAME" main | tee /etc/apt/sources.list.d/vim-ppa.list && \
                apt-key  adv --keyserver hkps://keyserver.ubuntu.com --recv-key 4AB0F789CBA31744CC7DA76A8CF63AD3F06FC659 && \
                apt-get update --fix-missing && \
                env DEBIAN_FRONTEND=noninteractive apt-get dist-upgrade --autoremove --purge --no-install-recommends -y \
                        build-essential \
                        bzip2 \
                        ca-certificates \
                        curl \
                        git \
                        libcanberra-gtk-module \
                        libgtk2.0-0 \
                        libx11-6 \
                        sudo \
                        graphviz \
                        vim-nox

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
RUN apt-get update -y

# Install miniconda - optinal
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN apt-get install -y software-properties-common
RUN apt-add-repository ppa:deadsnakes/ppa

RUN apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

RUN pip install torch torchaudio pandas soundfile editdistance scikit-learn packaging mkl-devel mkl jupyter

################### fairseq dependency ################
RUN apt-get update
RUN apt-get install -y cmake
RUN apt-get install -y build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev
RUN apt-get install -y libopenblas-dev
RUN apt-get install -y fftw3
RUN apt-get install -y libfftw3-dev
RUN apt-get update && apt-get install -y git libsndfile-dev && apt-get clean
#######################################################

RUN chmod -R 777 /usr/local/lib
RUN chmod -R 777 /usr/local/include
