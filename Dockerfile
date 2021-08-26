FROM tensorflow/tensorflow:1.15.4-gpu-py3
MAINTAINER Ben Shealy <btsheal@g.clemson.edu>

# install package dependencies
ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update -qq \
    && apt-get install -qq -y git python3-pip

# install python dependencies
RUN pip3 install -q matplotlib numpy pandas scikit-learn seaborn

# install tspg
WORKDIR /opt
RUN git clone -q https://github.com/ctargon/tspg.git

ENV TSPG_PATH "/opt/tspg"
ENV PYTHONPATH "${PYTHONPATH}:${TSPG_PATH}"

# initialize default work directory
WORKDIR /workspace
