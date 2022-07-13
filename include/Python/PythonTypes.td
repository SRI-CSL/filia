// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MLIR_DIALECT_PYTHON_IR_PYTHONTYPES
#define MLIR_DIALECT_PYTHON_IR_PYTHONTYPES

include "Python/PythonDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Python_Type<string name, string typeMnemonic>
    : TypeDef<Python_Dialect, name, [], "::mlir::python::PythonType"> {
  let mnemonic = typeMnemonic;
}

def Python_Value : Python_Type<"Value", "value"> {
  let summary = "A value in the Python language.";
  let description = [{
    This denotes a Python variable value.  A value is immutable, but if it is
    a class, then it's fields can be changed.
  }];
}

def Python_Cell : Python_Type<"Cell", "cell"> {
  let summary = "A cell for storing a Python value.";
  let description = [{
    This is a mutable cell that stores a Python value or may be unassigned.
  }];
}

def Python_Scope : Python_Type<"Scope", "scope"> {
  let summary = "Python scope";
  let description = [{
    A scope represents a mapping from Python variable names to values.
  }];
}

def Python_ReturnValue : Python_Type<"ReturnValue", "return_value"> {
  let summary = "Python return value";
  let description = [{
    This is a Python value or an exception (which should derive from BaseException).
    It is used for returning from Python functions.
  }];
}

#endif // MLIR_DIALECT_PYTHON_IR_PYTHONTYPES
