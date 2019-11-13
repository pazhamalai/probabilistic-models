# docker build -t gitlab.lrz.de:5005/i7/probabilistic-models:$(sha256sum < Dockerfile | cut -c -8) --compress - < Dockerfile

FROM openjdk:12-alpine

# Add build dependencies for PRISM
RUN apk --update --no-cache add \
    autoconf \
    automake \
    file \
    g++ \
    gcc \
    libtool \
    make \
    patch \
    python \
    subversion \
    swig \
    unzip \
    wget \
    git \
    openssh
