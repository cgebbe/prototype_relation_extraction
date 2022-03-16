# FROM gcr.io/kaggle-gpu-images/python:v111  # kaggle docker
FROM huggingface/transformers-pytorch-gpu:4.9.1 

# git and git-lfs. "git lfs install" needs to be run from inside repo
RUN apt update
RUN apt install git-lfs

# linting
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip install black

# relevant tools for docker image (in this cas for huggingface)
RUN pip install datasets seqeval tensorboard

# Problem: vscode tries to install stuff in /root (=default HOME) without permissions.
# Solution: setup different user and select new HOME. 
# https://github.com/microsoft/vscode-remote-release/issues/22#issuecomment-488843424
ARG USERNAME=myuser
RUN useradd -m $USERNAME
ENV HOME /home/$USERNAME