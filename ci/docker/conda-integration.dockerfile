# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

ARG repo
ARG arch=amd64
FROM ${repo}:${arch}-conda-cpp

ARG arch=amd64
ARG maven=3.5
ARG node=14
ARG jdk=8
ARG go=1.15

# Uninstall unused space-consuming packages
# (XXX: it would be better not to install them, but they are used by other
#  builds which are also based on conda-cpp)
RUN conda uninstall -q clangdev llvmdev valgrind

# Install Archery and integration dependencies
COPY ci/conda_env_archery.yml /arrow/ci/
RUN conda install -q \
        --file arrow/ci/conda_env_archery.yml \
        numpy \
        maven=${maven} \
        nodejs=${node} \
        openjdk=${jdk} && \
    conda clean --all --force-pkgs-dirs

# Install Rust and remove space-consuming docs
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    rm -rf `find /root/.rustup/ -name html -type d`

ENV GOROOT=/opt/go \
    GOBIN=/opt/go/bin \
    GOPATH=/go \
    PATH=/opt/go/bin:$PATH
RUN wget -nv -O - https://dl.google.com/go/go${go}.linux-${arch}.tar.gz | tar -xzf - -C /opt
