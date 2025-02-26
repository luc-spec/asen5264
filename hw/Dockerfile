FROM docker.io/debian:12-slim
USER root
RUN apt-get update -y && apt-get install -y git julia && rm -rf /etc/apt/lists/*
RUN git config --global user.name "nobody"
RUN git config --global user.email "nobody@no.email"
RUN julia -e 'using Pkg; Pkg.add(url="https://github.com/zsundberg/DMUStudent.jl")'
