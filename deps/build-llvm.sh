#!/bin/sh

cmake -S llvm-project/llvm -B llvm-build -G Ninja \
      -DCMAKE_INSTALL_PREFIX=$PWD/llvm-install \
      -DLLVM_ENABLE_PROJECTS=mlir \
      -DLLVM_INSTALL_UTILS=ON \
      -DLLVM_TARGETS_TO_BUILD="X86" \
      -DMLIR_ENABLE_BINDINGS_PYTHON=On