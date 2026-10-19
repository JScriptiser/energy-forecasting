FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib

# LightGBM wheels link against the system OpenMP runtime. The base image is pinned by digest;
# packages are upgraded and left unpinned on purpose to receive Debian security updates.
# hadolint ignore=DL3005,DL3008
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pip is only needed to install the hash-pinned requirements; it is removed from the runtime image
COPY requirements-dev.txt ./
RUN pip install --require-hashes --no-deps --root-user-action=ignore -r requirements-dev.txt \
 && pip uninstall -y --root-user-action=ignore pip

COPY pyproject.toml ./
COPY co2_forecast ./co2_forecast
COPY tests ./tests

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
USER 10001

# a batch job, not a service: nothing to health-check
HEALTHCHECK NONE

ENTRYPOINT ["python", "-m", "co2_forecast"]
