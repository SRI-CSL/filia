# LMC

LMC aims to provide cross-language program verification and bug finding
capabilities using the MLIR compiler infrastructure.  MLIR is a general
highly-extensible program intermediate representation suited for representing
code writen in a variety of languages.

LMC currently includes: (1) a Python to MLIR translation tool called py2mlir; (2)
a Javascript to MLIR translator called js2mlir; (3) a tool for running optimization
passes on Python called `python-opt`; and (4) a tool that runs standard MLIR optimization
passes that supports the Python and Javascript dialects called `script-opt`.


## Building

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run

```sh
cmake -S . -B build/debug -DMLIR_DIR=$PREFIX/lib/cmake/mlir
cmake --build build/debug --target check-python
```

To build the documentation from the TableGen description of the dialect operations, run

```sh
cmake --build build/debug --target mlir-doc
```

**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.
