FROM mambaorg/micromamba:git-79bf814-cuda12.4.1-ubuntu22.04

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt .

USER root

RUN apt-get -y update && apt-get -y install git

# pin python in base, make sure pip is there
RUN micromamba install -y -n base python=3.10 pip \
    && micromamba clean -afy

# makes subsequent RUNs use base explicitly instead of relying on PATH
ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN pip install --no-cache-dir -r requirements.txt \
      -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
