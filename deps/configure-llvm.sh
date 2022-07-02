#!/bin/sh

cmake -G Ninja \
    -S deps/llvm-project/llvm \
    -B deps/llvm-build \
    -DCMAKE_INSTALL_PREFIX=deps/llvm-install \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DMLIR_ENABLE_BINDINGS_PYTHON=On \
    -DLLVM_BUILD_TOOLS=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_TARGETS_TO_BUILD="X86" \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_USE_LINKER=lld