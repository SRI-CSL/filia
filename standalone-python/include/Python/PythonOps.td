// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef PYTHON_OPS
#define PYTHON_OPS

include "Python/PythonDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Python_UndefinedOp : Python_Op<"undefined", [NoSideEffect]> {
    let summary = "Introduces an undefined Python value.";
    let description = [{
        The `python.undefined` uses an operation trait to declare that it
        has no side effects.

        This operation returns a value.

        Example:

        ```mlir
        %0 = python.undefined
        ```
    }];

    let results = (outs ValueType:$result);
    let assemblyFormat = "attr-dict `:` type($result)";
}

#endif // PYTHON_OPS
