#!/bin/bash
set -ex

echo "Build"
cmake --build build/debug --target install

MLIRFILE=example/md5.mlir

echo "Generating $MLIRFILE"
./dist/debug/bin/py2mlir.py example/md5_example.py > $MLIRFILE

echo "Running python-opt"
OPTMLIRFILE=example/md5_opt.mlir
./dist/debug/bin/script-opt --python-load-store $MLIRFILE > $OPTMLIRFILE
./dist/debug/bin/script-opt -canonicalize $OPTMLIRFILE > example/md5_opt2.mlir

echo "Complete"