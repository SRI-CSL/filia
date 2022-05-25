#!/bin/sh
mkdir -p build/debug
cd build/debug
cmake -G Ninja ../.. -DCMAKE_BUILD_TYPE=Debug -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit python-opt