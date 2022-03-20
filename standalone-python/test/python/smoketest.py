# RUN: %PYTHON %s | FileCheck %s

from mlir_python.ir import *
from mlir_python.dialects import (
  builtin as builtin_d,
  python as python_d
)

with Context():
  python_d.register_dialect()
  module = Module.parse("""
    %0 = arith.constant 2 : i32
    %1 = python.foo %0 : i32
    """)
  # CHECK: %[[C:.*]] = arith.constant 2 : i32
  # CHECK: python.foo %[[C]] : i32
  print(str(module))
