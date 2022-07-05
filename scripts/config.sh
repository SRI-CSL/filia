#!/bin/sh
#mkdir -p build/debug

# -B build/debug -G Ninja

PREFIX=../../deps/llvm-install
cmake -S ../..  -DCMAKE_BUILD_TYPE=Debug  -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit