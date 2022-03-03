#!/bin/bash
set -e

npm run build
pushd mlir-loader > /dev/null
make
popd > /dev/null

node ./dist/app.js script ./dist/app.js app.mlir
./mlir-loader/loadmlir app.mlir