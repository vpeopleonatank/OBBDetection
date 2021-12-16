# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.7    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM ubuntu:18.04
ENV LANG C.UTF-8
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
  PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
  GIT_CLONE="git clone --depth 10" && \
  rm -rf /var/lib/apt/lists/* \
  /etc/apt/sources.list.d/cuda.list \
  /etc/apt/sources.list.d/nvidia-ml.list && \
  apt-get update && \
  # ==================================================================
  # tools
  # ------------------------------------------------------------------
  DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
  build-essential \
  apt-utils \
  ca-certificates \
  wget \
  git \
  vim \
  libssl-dev \
  curl \
  unzip \
  unrar \
  && \
  $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
  cd ~/cmake && \
  ./bootstrap && \
  make -j"$(nproc)" install && \
  # ==================================================================
  # python
  # ------------------------------------------------------------------
  DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
  software-properties-common \
  && \
  add-apt-repository ppa:deadsnakes/ppa && \
  apt-get update && \
  DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
  python3.7 \
  python3.7-dev \
  python3-distutils-extra \
  && \
  wget -O ~/get-pip.py \
  https://bootstrap.pypa.io/get-pip.py && \
  python3.7 ~/get-pip.py && \
  ln -s /usr/bin/python3.7 /usr/local/bin/python3 && \
  ln -s /usr/bin/python3.7 /usr/local/bin/python && \
  $PIP_INSTALL \
  setuptools \
  && \
  $PIP_INSTALL \
  numpy \
  scipy \
  pandas \
  cloudpickle \
  scikit-image>=0.14.2 \
  scikit-learn \
  matplotlib \
  Cython \
  tqdm \
  && \
  # ==================================================================
  # pytorch
  # ------------------------------------------------------------------
  $PIP_INSTALL \
  future \
  numpy \
  protobuf \
  enum34 \
  pyyaml \
  typing 
  # && \
RUN PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
  $PIP_INSTALL torch==1.6.0+cpu torchvision==0.7.0+cpu -f \
  https://download.pytorch.org/whl/torch_stable.html \
  && \
  # ==================================================================
  # config & cleanup
  # ------------------------------------------------------------------
  ldconfig && \
  apt-get clean && \
  apt-get autoremove && \
  rm -rf /var/lib/apt/lists/* /tmp/* ~/*


RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.6.0/index.html
# RUN pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

RUN git clone https://github.com/vpeopleonatank/OBBDetection.git --recursive /obbdetection
WORKDIR /obbdetection
RUN git checkout dev
# ENV FORCE_CUDA="1"
# RUN cd BboxToolkit
WORKDIR /obbdetection/BboxToolkit
RUN python setup.py develop
# RUN cd ..
WORKDIR /obbdetection
RUN pip install cython numpy seaborn --no-cache-dir
RUN pip install mmpycocotools

RUN pip install -v -e .

