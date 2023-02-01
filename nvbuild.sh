#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Configure, build, and install Jax
# The default values are set for development, not the CI.
#
# If the directory /cache/ exist, we tell bazel to use it as a disk
# cache. This way when developing, we can just mount a directory there
# and it the compilation will be cached there.

# Exit at error
set -ex

export TEST_TMPDIR=/tmp/bazel_cache

Usage() {
  set +x
  echo "Configure, build, and install JAX and Jaxlib"
  echo ""
  echo "  Usage: $0 [OPTIONS]"
  echo ""
  echo "    OPTIONS                        DESCRIPTION"
  echo "    --clean                        Delete local configuration and bazel cache"
  echo "    --no_clean                     Do not delete local configuration and bazel cache (default)"
  echo "    --sm SM1,SM2,...               The SM to use to compile TF"
  echo "    --sm local                     Query the SM of available GPUs (default)"
  echo "    --sm all                       All current SM"
  echo "    --dbg                          Build in debug mode"
  echo "    --jaxlib_only                  Only build and install jaxlib"
  echo "    --build_param PARAM            Param passed to the jaxlib build command. Can be passed many times."
  echo "                                   If you want to pass a bazel parameter, you must do it like this:"
  echo "                                       --build_param=--bazel_options=..."
  set -x
}

CLEAN=0
DBG=0
JAXLIB_ONLY=0
PARAM=""
TFDIR="/opt/jax/tensorflow-source"
export TF_CUDA_COMPUTE_CAPABILITIES="local"


while [[ $# -gt 0 ]]; do
  case $1 in
    "--help"|"-h")   Usage; exit 1 ;;
    "--clean")       CLEAN=1 ;;
    "--no_clean")    CLEAN=0 ;;
    "--dbg")         DBG=1 ;;
    "--jaxlib_only") JAXLIB_ONLY=1 ;;
    "--sm")          shift 1;
                     TF_CUDA_COMPUTE_CAPABILITIES=$1
                     ;;
    "--tf_dir")      shift 1;
                     TFDIR="$1"
                     ;;
    "--build_param") shift 1;
                     PARAM="$PARAM $1"
                     ;;
    *)
      echo UNKNOWN OPTION $1
      echo Run $0 -h for help
      exit 1
  esac
  shift 1
done


if [[ -z "${TARGETARCH}" ]]; then
  ARCH=$(uname -m)
  if [[ "$ARCH" == "x86_64" ]]; then
    TARGETARCH="amd64"
  elif [[ "$ARCH" == "aarch64" ]]; then
    TARGETARCH="arm64"
    # Avoid ABORT failures on SBSA builders due to insufficeint memory.
    N_CPU_CORES=$(grep -c ^processor /proc/cpuinfo)
    [[ $N_CPU_CORES -gt 40 ]] && MAX_BUILD_JOBS=40 || MAX_BUILD_JOBS=$N_CPU_CORES
  else
    echo Unknown arch $ARCH
    exit 1
  fi
fi

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
pushd "${TFDIR}"
if [[ "$TF_CUDA_COMPUTE_CAPABILITIES" == "all" ]]; then
  TF_CUDA_COMPUTE_CAPABILITIES="$(${THIS_DIR}/nvarch.sh ${TARGETARCH})"
  if [[ $? -ne 0 ]]; then exit 1; fi
elif [ "$TF_CUDA_COMPUTE_CAPABILITIES" == "local" ]; then
echo DISCOVERING LOCAL COMPUTE CAPABILITIES
set +e # Allow errors so that a.out can be cleaned up
TF_CUDA_COMPUTE_CAPABILITIES=$( \
cat <<EOF | nvcc -x c++ --run -
#include <stdio.h>
#include <string>
#include <set>
#include <cuda_runtime.h>
#define CK(cmd) do {                    \
  cudaError_t r = (cmd);                \
  if (r != cudaSuccess) {               \
    fprintf(stderr,                     \
            "CUDA Runtime error: %s\n", \
            cudaGetErrorString(r));     \
    exit(EXIT_FAILURE);                 \
  }                                     \
 } while (false)
using namespace std;
int main() {
  int device_count;
  CK(cudaGetDeviceCount(&device_count));
  set<string> set;
  for(int i=0; i<device_count; i++) {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, i));
    set.insert(to_string(prop.major)+"."+to_string(prop.minor));
  }
  int nb_printed = 0;
  for(string sm: set) {
    if (nb_printed > 0) printf(",");
    printf("%s", sm.data());
    ++nb_printed;
  }
  printf("\n");
}
EOF
)
R=$?
rm a.out
if [[ "$R" -ne 0 ]]; then
  exit 1
fi
set -e
fi

echo "CUDA COMPUTE: ${TF_CUDA_COMPUTE_CAPABILITIES}"

export TF_NEED_CUDA=1
export TF_NEED_CUTENSOR=1
export TF_NEED_TENSORRT=1
export TF_CUDA_PATHS=/usr,/usr/local/cuda
export TF_CUDNN_PATHS=/usr/lib/$(uname -m)-linux-gnu
export TF_CUDA_VERSION=$(ls /usr/local/cuda/lib64/libcudart.so.*.*.* | cut -d . -f 3-4)
export TF_CUBLAS_VERSION=$(ls /usr/local/cuda/lib64/libcublas.so.*.*.* | cut -d . -f 3)
export TF_CUDNN_VERSION=$(echo "${CUDNN_VERSION}" | cut -d . -f 1)
export TF_NCCL_VERSION=$(echo "${NCCL_VERSION}" | cut -d . -f 1)

if [ "${TARGETARCH}" = "amd64" ] ; then export CC_OPT_FLAGS="-march=sandybridge -mtune=broadwell" ; fi
if [ "${TARGETARCH}" = "arm64" ] ; then export CC_OPT_FLAGS="-march=armv8-a" ; fi

popd

#cd jax-source

# Uninstall them in case this script was used before.
if [ "$EUID" -ne 0 ]; then
    SUDO=sudo
fi

if [ "$JAXLIB_ONLY" == "0" ]; then
    ${SUDO} pip uninstall -y jax jaxlib
else
    ${SUDO} pip uninstall -y jaxlib
fi


if [ -d /cache ]; then
    PARAM="${PARAM} --bazel_options=--disk_cache=/cache"
fi
if [ "$DBG" == "1" ]; then
    PARAM="${PARAM} --bazel_options=-c --bazel_options=dbg --bazel_options=--strip=never --bazel_options=--cxxopt=-g --bazel_options=--cxxopt=-O0"
fi

echo "TFDIR:" ${TFDIR}
python build/build.py --enable_cuda --cuda_path=$TF_CUDA_PATHS --cudnn_path=$TF_CUDNN_PATHS \
                --cuda_version=$TF_CUDA_VERSION --cudnn_version=$TF_CUDNN_VERSION \
                --cuda_compute_capabilities=$TF_CUDA_COMPUTE_CAPABILITIES \
                --enable_nccl=true --bazel_path=/usr/local/bin/bazel \
                --bazel_options=--override_repository=org_tensorflow=${TFDIR} \
                $PARAM
                # Disable Bazel cache server until we troubleshoot why it breaks jaxlib builds
                # $(/opt/jax/nvbazelcache)
pip --disable-pip-version-check install --force-reinstall dist/*.whl
if [ "$JAXLIB_ONLY" == "0" ]; then
    pip --disable-pip-version-check install .
fi

mv build/bazel* bazel && chmod 0755 bazel && ./bazel && \
install bazel /usr/local/bin/

if [ "$CLEAN" == "1" ]; then
    rm -rf dist/
    rm -rf bazel
    bazel clean --expunge
    rm .jax_configure.bazelrc
    rm -rf ${HOME}/.cache/bazel /tmp/*
fi
