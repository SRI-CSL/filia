#!/bin/bash
set -e

npm run build
node ./dist/app.js script ../standalone-python/example/insecure_eval.js app.mlir
