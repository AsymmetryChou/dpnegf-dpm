FROM ubuntu:20.04
SHELL ["/bin/bash", "-c"]

ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=23.11.0-0
ARG TARGETPLATFORM

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN apt-get update > /dev/null && \
    apt-get install --no-install-recommends --yes \
        wget bzip2 ca-certificates \
        git \
        tini \
        g++ \
        > /dev/null && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes  && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc

WORKDIR /app
COPY . .

# 2. Create the environment and install all dependencies
RUN \
    sed -i 's/build-backend = "poetry_dynamic_versioning.backend"/build-backend = "poetry.core.masonry.api"/' pyproject.toml && \
    conda create -n dpnegf python=3.10 -c conda-forge -y && \
    git clone https://github.com/deepmodeling/DeePTB.git && \
    conda run -n dpnegf pip install --upgrade pip setuptools wheel && \
    # [1] Force the CPU-only PyTorch 2.5.1 wheel to shrink the image and align the low-level ABI.
    conda run -n dpnegf pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    # [2] Pull torch-scatter from the PyG wheel index and use --only-binary=torch-scatter to fully disable source builds.
    # If no matching wheel is found it fails immediately instead of spending ~10 minutes compiling a package that would crash at runtime.
    conda run -n dpnegf pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cpu.html --only-binary=torch-scatter && \
    # [3] Guard the local-repo installs with the CPU index so hidden dependencies can't replace the CPU torch with a CUDA build.
    conda run -n dpnegf pip install ./DeePTB torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cpu && \
    conda run -n dpnegf pip install ./ torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cpu && \
    conda clean --all -y && \
    rm -rf /root/.cache/pip

# 3. Activate the dpnegf environment by default in interactive shells
RUN echo "conda activate dpnegf" >> ~/.bashrc
