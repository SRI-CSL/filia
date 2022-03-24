#!/bin/sh



PYTHONPATH="build/debug/python_packages/python" python ./genmlir/genmlir.py "example/insecure_eval.py" > example/insecure_eval_python.mlir
./build/debug/bin/script-opt -canonicalize ./example/insecure_eval_python.mlir > ./example/insecure_eval_python_can.mlir
# Python specific optimizations
./build/debug/bin/python-opt ./example/insecure_eval_python.mlir 2> example/insecure_eval_python_opt.mlir
# Canonicalize after Python specific optimization
./build/debug/bin/script-opt -canonicalize ./example/insecure_eval_python_opt.mlir > ./example/insecure_eval_python_opt_can.mlir
