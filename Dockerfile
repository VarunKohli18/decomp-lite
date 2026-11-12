# Everything decomp-lite needs: an x86-64 Linux gcc + binutils, Ghidra, Python.
#
#   docker build --platform linux/amd64 -t decomp-lite .
#   docker run --platform linux/amd64 --rm -it --network host -v "$PWD":/work decomp-lite \
#       python3 -m decomplite benchmark --config benchmark.yaml --phase build
#
# Always build and run as linux/amd64: Ghidra ships its native decompiler
# prebuilt for x86-64 Linux only. On an x86-64 host the flag is a no-op; on
# Apple Silicon it selects emulation (slower, same output).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    GHIDRA_INSTALL_DIR=/root/opt/ghidra \
    GHIDRA_VERSION=11.3.1

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libc6-dev binutils \
        openjdk-21-jdk-headless \
        python3 python3-pip python3-yaml python3-requests python3-pyelftools python3-pytest \
        curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --break-system-packages --ignore-installed idna \
        "tree-sitter>=0.22,<0.23" "tree-sitter-c==0.21.4" "anthropic>=2"

COPY scripts/install_ghidra.sh /tmp/install_ghidra.sh
RUN bash /tmp/install_ghidra.sh "${GHIDRA_VERSION}" && rm /tmp/install_ghidra.sh

# The repo is bind-mounted here and run in place (`python3 -m decomplite`), so
# data/, results/ and reports/ land on the host.
WORKDIR /work
CMD ["bash"]
