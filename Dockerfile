FROM mambaorg/micromamba:2.0.5

WORKDIR /app

# Install the dependencies
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Activate the environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1

COPY pyproject.toml README.md /app/
COPY data/ /app/data
COPY --chown=$MAMBA_USER:$MAMBA_USER experiments/ /app/experiments
COPY --chown=$MAMBA_USER:$MAMBA_USER src/dli/ /app/src/dli

RUN pip install --no-cache-dir -e /app

CMD ["/bin/bash"]
