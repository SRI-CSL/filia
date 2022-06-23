#!/bin/bash
set -e

echo "Build"
cmake --build build/debug

MLIRFILE=example/md5.mlir

echo "Generating $MLIRFILE"
PYTHONPATH="build/debug/python_packages/python" python genmlir/genmlir.py example/md5_example.py > $MLIRFILE

echo "Running python-opt"
OPTMLIRFILE=example/md5_opt.mlir
./build/debug/bin/python-opt --verify $MLIRFILE > $OPTMLIRFILE
./build/debug/bin/script-opt -canonicalize $OPTMLIRFILE >example/md5_opt2.mlir

echo "Complete"