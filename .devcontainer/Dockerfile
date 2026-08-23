# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Prevent interactive prompts during package installation and configure Python environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies and clean up apt cache in a single layer to minimize image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    openmpi-bin \
    libopenmpi-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory for devcontainer workspace
WORKDIR /workspace
