# An out-of-tree MLIR dialect

This provides Python and Javascript [MLIR](https://mlir.llvm.org/) dialects along with a standalone `opt`-like tool to operate on those dialect.

## Building

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
cmake -S . -B build/debug -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build build/debug --target check-python
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build build/debug --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

