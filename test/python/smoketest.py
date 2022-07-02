# RUN: %PYTHON %s | FileCheck %s

from mlir_python.ir import *
from mlir_python.dialects import (
  builtin as builtin_d,
  python as python_d
)

with Context():
  python_d.register_dialect()
  module = Module.parse("""
    %0 = python.undefined : !python.value
    """)
  # CHECK: python.undefined : !python.value
  print(str(module))
