# hash:sha256:31ecb6f5f67584605c1f3aabfe7397e6e936d20541f4fcab2ee324433d52e850
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-toolkit-11-0=11.0.3-1 \
        pkg-config=0.29.1-0ubuntu2 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        jupyter==1.0.0 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    pandas==1.1.5

COPY postInstall /
RUN /postInstall
