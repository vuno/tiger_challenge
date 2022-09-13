FROM nvidia/cuda:11.1-devel-ubuntu20.04

# Sync local time to Netherlands (location of organizers)
ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python3.8
RUN : \
    && apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && apt-get install -y --no-install-recommends python3.8-venv \
    && apt-get install libpython3.8-dev -y \
    && apt-get clean \
    && :

# Add env to PATH
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install ASAP
RUN : \
    && apt-get update \
    && apt-get -y install curl \
    && curl --remote-name --location "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1-(Nightly)/ASAP-2.1-Ubuntu2004.deb" \
    && dpkg --install ASAP-2.1-Ubuntu2004.deb || true \
    && apt-get -f install --fix-missing --fix-broken --assume-yes \
    && ldconfig -v \
    && apt-get clean \
    && echo "/opt/ASAP/bin" > /venv/lib/python3.8/site-packages/asap.pth \
    && rm ASAP-2.1-Ubuntu2004.deb \
    && :

# Install python package: wheel
RUN pip install wheel==0.37.0

# Install python packages: pytorch, torchvision
RUN : \
    && pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Copy challenge codes
# TODO: modify when moved as root (at official repo)
ARG PROJECT_FOLDER=/vuno
COPY ./ $PROJECT_FOLDER

# Install python requirements
RUN :\
    && pip install --upgrade pip \
    && pip install --upgrade setuptools \
    && python -m pip install -r $PROJECT_FOLDER/requirements.txt

# Make user
ARG UID
RUN groupadd -r user && useradd -u $UID --create-home -r -g user user
RUN chown user /home/user
RUN mkdir /output/
RUN chown user /output/
USER user

WORKDIR $PROJECT_FOLDER

# Set python environment
ENV PYTHONPATH "/vuno"

# Cmd and entrypoint
CMD ["algorithm"]
ENTRYPOINT ["python"]

# Compute requirements
LABEL processor.cpus="1"
LABEL processor.cpu.capabilities="null"
LABEL processor.memory="15G"
LABEL processor.gpu_count="1"
LABEL processor.gpu.compute_capability="null"
LABEL processor.gpu.memory="11G"
