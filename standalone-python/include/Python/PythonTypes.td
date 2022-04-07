// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MLIR_DIALECT_PYTHON_IR_PYTHONTYPES
#define MLIR_DIALECT_PYTHON_IR_PYTHONTYPES

include "Python/PythonDialect.td"

class Python_Type<string name, string typeMnemonic>
    : TypeDef<Python_Dialect, name, [], "::mlir::python::PythonType"> {
  let mnemonic = typeMnemonic;
}

def Python_Value : Python_Type<"Value", "value"> {
  let summary = "Python value";
  let description = [{
    This type represents a Python variable value.
  }];
}

def Python_Scope : Python_Type<"Scope", "scope"> {
  let summary = "A scope maps names to the values.";
  let description = [{
    This type represents a mapping from Python variable names to values.
  }];
}

#endif // MLIR_DIALECT_PYTHON_IR_PYTHONTYPES
