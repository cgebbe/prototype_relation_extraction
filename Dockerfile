# FROM gcr.io/kaggle-gpu-images/python:v111  # kaggle docker
FROM huggingface/transformers-pytorch-gpu:4.9.1 

# ========================================
# install linux packages
# ========================================
# git and git-lfs. "git lfs install" needs to be run from inside repo
RUN apt update
RUN apt install -y git-lfs vim

# ========================================
# install pip packages
# ========================================
RUN python3 -m pip install --upgrade pip setuptools wheel

# fixes bug when installing click-package,
# see https://stackoverflow.com/a/37868546/2135504
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY requirements.txt .
RUN pip install -r requirements.txt

# ========================================
# Workarounds
# ========================================
# Problem: vscode tries to install stuff in /root (=default HOME) without permissions.
# Solution: setup different user and select new HOME. 
# https://github.com/microsoft/vscode-remote-release/issues/22#issuecomment-488843424
ARG USERNAME=myuser
RUN useradd -m $USERNAME
ENV HOME /home/$USERNAME