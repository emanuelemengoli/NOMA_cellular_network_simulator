# Use the official Ubuntu 22.04 as a base image
FROM ubuntu:22.04

# Set environment variable to non-interactive to avoid some installation prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    gpg \
    nano \
    python3 \
    python3-pip \
    && apt-get clean

RUN apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev

RUN apt-get upgrade -y

# Confirm Python and pip installations
RUN python3 --version && \
    pip --version

COPY container_install.py .
RUN python3 container_install.py

# # Set the default command to run when starting the container
CMD [ "/bin/bash" ]

# Set the default command to keep the container running
# CMD [ "tail", "-f", "/dev/null" ]
# ENTRYPOINT ["tail", "-f", "/dev/null"]