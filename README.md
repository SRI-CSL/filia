# Filia

Filia is a project to build multi-language program verification and bug finding
capabilities using the MLIR compiler infrastructure.  MLIR is a general
highly-extensible program intermediate representation suited for representing
code writen in a variety of languages.

Filia currently includes: (1) `py2mlir` -- a Python to MLIR translation tool;
(2) `js2mlir` -- a Javascript to MLIR translator; (3) `script-opt`-- an
optimization tool that supports the Python and Javascript dialects.

`js2mlir` is self-contained and independent of the other components.
See `js2mlir/README.md` for instructions on building and using `js2mlir`.


## Building py2mlir and script-opt

Filia requires a build of LLVM including MLIR and the MLIR Python bindings.
We include a submodule that links to the version of LLVM used for testing Filia.

### Building MLIR and Python bindings

Filia requires a build of MLIR with the Python bindings.  As these are not
yet shipped with system versions of LLVM, we recommend that you clone the
LLVM submodule and build this version.  We also provide a script to configure
LLVM with MLIR enabled.  To use it, you need to first ensure that
[`cmake`](https://cmake.org/), [`ninja`](https://ninja-build.org/) and
[`python`](https://www.python.org/) with the PIP package manager are installed.
Once these are installed, run the following commands from the Filia root
directory:

```
pip install -r deps/llvm-project/mlir/python/requirements.txt
./deps/configure-llvm.sh
cmake --build deps/llvm-build --target install
```

This will install LLVM into the directory `deps/llvm-install`.

### Building Filia

Once LLVM is installed, you can configure and build Filia with the following commands:

```sh
cmake -S . -B build/debug -DMLIR_DIR=deps/llvm-install/lib/cmake/mlir
cmake --build build/debug --target install
```

## Using Filia

Fiia is still in active development and only supports a small subset of
Python to see it working on an MD5 example, you can run the following commands:

```
./dist/debug/bin/py2mlir.py example/md5_example.py > md5.mlir
```

This MLIR file can then be optimized by `script-opt`.  For example,

```
./dist/debug/bin/script-opt --python-load-store md5.mlir > md5-opt.mlir
```