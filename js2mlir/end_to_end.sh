#!/bin/bash
set -e

npm run build
./bin/js2mlir script ../standalone-python/example/insecure_eval.js app.mlir
