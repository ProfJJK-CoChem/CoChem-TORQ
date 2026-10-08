# syntax=docker/dockerfile:1
# Separate, redistributable Linux CPU calculation profile. Qualification is
# recorded against the built image ID/digest after a genuine calculation.
FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    XDG_STATE_HOME=/tmp/cochem-state
RUN groupadd --gid 1000 student && useradd --uid 1000 --gid 1000 --create-home student \
    && python -m venv /opt/engine-env && mkdir -p /opt/torq-source /work \
    && chown -R student:student /opt/engine-env /opt/torq-source /work
ENV PATH="/opt/engine-env/bin:${PATH}"
WORKDIR /opt/torq-source
COPY --chown=student:student . .
USER student
RUN --mount=type=secret,id=proxy_ca,uid=1000,gid=1000,mode=0444 \
    if [ -r /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    python -m pip install --require-hashes -r ci_tools/requirements-pyscf-py312.txt \
    && python -m pip install --no-deps . \
    && python -m pip check
WORKDIR /work
ENTRYPOINT ["cochem-torq"]
