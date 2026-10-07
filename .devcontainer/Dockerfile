# syntax=docker/dockerfile:1
FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

# Prevent interactive prompts during package installation and configure Python environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies and clean up apt cache in a single layer to minimize image size
RUN --mount=type=secret,id=proxy_ca \
    if [ -r /run/secrets/proxy_ca ]; then \
      export APT_CONFIG=/tmp/torq-apt.conf; \
      echo 'Acquire::https::CaInfo "/run/secrets/proxy_ca";' > "$APT_CONFIG"; \
    fi; \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    gh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/torq-apt.conf

# Give the editor user an owned virtual environment; no root pip installs.
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} vscode \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash vscode \
    && python -m venv /opt/torq-venv \
    && mkdir -p /home/vscode/CoChem_Artifacts \
    && chown -R vscode:vscode /opt/torq-venv /home/vscode/CoChem_Artifacts

ENV PATH="/opt/torq-venv/bin:${PATH}"
WORKDIR /workspaces/CoChem-TORQ
USER vscode
