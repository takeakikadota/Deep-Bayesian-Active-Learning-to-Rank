FROM nvcr.io/nvidia/tensorflow:21.05-tf2-py3
USER root

# timezone
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install
RUN apt-get update \
    && apt-get install -y tzdata \
    && apt-get install -y --no-install-recommends \
        libsm6 \
        libxext6 \
        libgtk2.0-dev \
        libgl1-mesa-glx \
        git \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install -q \
        keras==2.4.3 \
        numpy==1.19.5 \
        pandas==1.1.5 \
        scipy==1.5.4 \
        opencv-python==4.5.1.48 \
        matplotlib==3.3.4 \
        seaborn==0.11.2 \
        tensorboard==2.4.0 \
        ipython==7.16.3 \
        scikit-learn==1.3.2 \
        h5py \
        Pillow \
        tqdm

# Create a non-root user matching the host UID/GID
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} dockeruser \
    && useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash dockeruser

USER dockeruser
WORKDIR /workdir