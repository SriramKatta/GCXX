ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG GCC_VERSION=13
ARG CREATE_RENDER_GROUP=false
ARG USER_GROUPS=sudo
ARG CONTAINER_NAME=gcxx-dev

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang-format \
        clang-tidy \
        clangd \
        cmake \
        cmake-format \
        curl \
        g++-${GCC_VERSION} \
        gcc-${GCC_VERSION} \
        gdb \
        git \
        less \
        ninja-build \
        pkg-config \
        sudo \
        vim \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} 100 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-${GCC_VERSION} 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-${GCC_VERSION} 100 \
    && rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc-${GCC_VERSION} \
    CXX=/usr/bin/g++-${GCC_VERSION} \
    CONTAINER_NAME=${CONTAINER_NAME}

RUN if [ "${CREATE_RENDER_GROUP}" = "true" ]; then getent group render || groupadd render; fi \
    && useradd --create-home --shell /bin/bash vscode \
    && usermod -aG "${USER_GROUPS}" vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

USER vscode
WORKDIR /workspaces/GCXX
