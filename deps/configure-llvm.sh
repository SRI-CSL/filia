#!/bin/bash
cd "$(dirname $0)/.."

cmake -G Ninja \
    -S deps/llvm-project/llvm \
    -B deps/llvm-build \
    -DCMAKE_INSTALL_PREFIX=deps/llvm-install \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_ENABLE_ZLIB=Off \
    -DMLIR_ENABLE_BINDINGS_PYTHON=On \
    -DLLVM_BUILD_TOOLS=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_TARGETS_TO_BUILD="X86" \
    ${EXTRA_ARGS:-}