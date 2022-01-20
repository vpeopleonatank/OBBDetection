ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
# RUN pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

RUN pip install pydantic fastapi python-multipart uvicorn

RUN conda clean --all
EXPOSE 8081

WORKDIR /obbdetection

COPY . /obbdetection

ENV FORCE_CUDA="1"
# RUN cd BboxToolkit
WORKDIR /obbdetection/BboxToolkit
RUN python setup.py develop
# RUN cd ..
WORKDIR /obbdetection
RUN pip install cython numpy seaborn --no-cache-dir
RUN pip install mmpycocotools

RUN pip install -v -e .

CMD ["python", "app.py"]
