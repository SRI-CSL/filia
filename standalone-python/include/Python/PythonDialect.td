// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef PYTHON_DIALECT
#define PYTHON_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Python dialect definition.
//===----------------------------------------------------------------------===//

def Python_Dialect : Dialect {
    let name ="python";
    let summary = "A MLIR dialect for Python.";
    let description = [{
        This dialect captures operations needed to encode Python programs
        in MLIR.
    }];
    let cppNamespace = "::mlir::python";
}


def PythonIsValueTypePred : CPred<"$_self.isa<ValueType>()">;
def ValueType : DialectType<Python_Dialect, PythonIsValueTypePred, "value">;

//===----------------------------------------------------------------------===//
// Base python operation definition.
//===----------------------------------------------------------------------===//

class Python_Op<string mnemonic, list<Trait> traits = []> :
        Op<Python_Dialect, mnemonic, traits>;

#endif // PYTHON_DIALECT
