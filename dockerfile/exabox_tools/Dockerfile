# Exabox operations image, published as ghcr.io/tenstorrent/tt-metal/exabox-tools.
#
# One image for everything that operates a BH Galaxy Exabox cluster from a container:
#   - tools/scaleout/exabox/recover.sh, run_validation.sh, mpi-docker
#   - build/tools/scaleout/run_cluster_validation (physical validation, link retraining)
#   - tools/scaleout/exabox/health_check_test_suite (fabric system health check)
#
# Content is the same as the upstream-tests images (full build tree, runtime/, ttnn wheel,
# source checkout at the build SHA). That is deliberate: run_cluster_validation links the
# whole stack (tt_metal, umd, the custom OpenMPI in /opt/openmpi-v5.0.7-ulfm, ...) and the
# health check imports ttnn, so this cannot be a minimal image like
# dockerfile/scaleout_cabling_generator/Dockerfile.
#
# What differs from dockerfile/upstream_test_images/Dockerfile.template:
#   - No test-script ENTRYPOINT. recover.sh, mpi-docker and the k8s jobs supply the command.
#   - Health-check Python requirements are baked in. Today the launchers install them at
#     container start over the network with `|| true`.
#   - OCI revision/version labels so a running container can be traced back to a commit.
#   - A build-time smoke gate as the last RUN. If a tool cannot start, the image is not pushed,
#     so :latest can only ever move to something that at least starts.
#
# Built and published by .github/workflows/exabox-tools-image.yaml (daily from main, or on
# dispatch). Its own workflow on purpose: upstream-tests fails often and :latest here must
# keep tracking main regardless.
#
# Build context (same layout as the build-images job in upstream-tests.yaml):
#   - the repo checkout
#   - _tt-metal/   : extracted ttm_any.tar.zst (build/, runtime/, ...)
#   - ttnn-*.whl   : the wheel artifact
#
# Consumers pin either :latest (moving, follows main) or the immutable git-describe tag.

# The dev image is pinned to ubuntu-22.04 like Dockerfile.template. The build artifact must
# come from the same platform (the workflow builds with platform "Ubuntu 22.04").
ARG TT_METAL_DEV_VERSION=latest
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:${TT_METAL_DEV_VERSION} AS base

ARG TT_METAL_COMMIT_SHA
ARG IMAGE_VERSION=unknown

ENV DEBIAN_FRONTEND=noninteractive

LABEL org.opencontainers.image.source="https://github.com/tenstorrent/tt-metal"
LABEL org.opencontainers.image.description="Exabox operations tools: recover.sh, cluster validation, fabric system health check"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.revision="${TT_METAL_COMMIT_SHA}"
LABEL org.opencontainers.image.version="${IMAGE_VERSION}"

USER user

# No apt here on purpose: build containers on the CI runners cannot reach archive.ubuntu.com
# (connection timeouts, verified twice), so pciutils for the health check's lspci stays a
# container-start install in launch-health-check.sh, as it is for the upstream-tests image.

RUN git clone https://github.com/tenstorrent/tt-metal --recurse-submodules /home/user/tt-metal

WORKDIR /home/user/tt-metal

RUN git reset --hard ${TT_METAL_COMMIT_SHA} && git submodule update --init --recursive

ENV PYTHONPATH=/home/user/tt-metal
ENV LD_LIBRARY_PATH=/home/user/tt-metal/build/lib
ENV TT_METAL_HOME=/home/user/tt-metal

COPY _tt-metal/build/ ./build/
COPY _tt-metal/runtime/ ./runtime/
COPY ttnn-*.whl ./

# UMD api_tests embed UMD_TESTS_ROOT_PATH=/work/tt_metal/third_party/umd/tests at compile time.
# The metal build runs at /work, but this image keeps sources under /home/user/tt-metal.
RUN sudo mkdir -p /work/tt_metal/third_party/umd && \
    sudo ln -sfn /home/user/tt-metal/tt_metal/third_party/umd/tests /work/tt_metal/third_party/umd/tests

RUN uv pip install $(ls -1 *.whl)
RUN uv pip install -r tools/scaleout/exabox/health_check_test_suite/test_infrastructure/requirements.txt

# Smoke gate. Everything here runs without hardware in seconds on the build runner.
#   - run_cluster_validation starts and has no unresolved shared libraries
#   - the CPU-only scaleout descriptor tests pass against the checked-out fixtures
#   - the operator scripts parse
#   - the health-check runner imports (pulls in its requirements) and prints usage
#   - ttnn imports
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN ./build/tools/scaleout/run_cluster_validation --help > /dev/null \
    && ldd ./build/tools/scaleout/run_cluster_validation | tee /tmp/ldd.txt \
    && ! grep -q "not found" /tmp/ldd.txt \
    && rm -f /tmp/ldd.txt \
    && TT_METAL_LOGGER_LEVEL=Fatal ./build/test/tools/scaleout/test_descriptor_merger \
    && TT_METAL_LOGGER_LEVEL=Fatal ./build/test/tools/scaleout/test_instance_filter \
    && bash -n tools/scaleout/exabox/recover.sh \
    && bash -n tools/scaleout/exabox/run_validation.sh \
    && bash -n tools/scaleout/exabox/mpi-docker \
    && python3 tools/scaleout/exabox/health_check_test_suite/test_infrastructure/run_health_check.py --help > /dev/null \
    && TRACY_NO_INVARIANT_CHECK=1 python3 -c 'import ttnn'

# Give others full access to the user directory (for cross-user MPI runs)
RUN sudo chmod -R 777 /home/user

# Allow files with insecure permissions to be used as authorized keys by disabling SSH StrictModes
# (for MPI operator key mounts)
RUN sudo mkdir -p /run/sshd
RUN echo "StrictModes no" | sudo tee -a /etc/ssh/sshd_config

# No entrypoint. The dev base image's /usr/local/bin/entrypoint.sh would otherwise be
# inherited; it runs `set -e; service ssh start; exec "$@"`, which fails noisily as a
# non-root user and would make every 'docker run <image> <cmd>' depend on sshd starting.
# Every consumer (recover.sh, mpi-docker, launch-health-check.sh, the k8s jobs) supplies
# its own command.
ENTRYPOINT []
CMD ["/bin/bash"]
