// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef JAVASCRIPT_DIALECT
#define JAVASCRIPT_DIALECT

include "mlir/IR/OpBase.td"

// JavaScipt dialect definition.
def JavaScript_Dialect : Dialect {
    let name ="js";
    let summary = "A dialect for JavaScript operations.";
    let description = [{
        This dialect captures operations needed to describe JavaScript
        types and functions.
    }];
    let cppNamespace = "::mlir::js";
    let useDefaultTypePrinterParser = 1;
}

// Base Javascript operation
class JavaScript_Op<string mnemonic, list<Trait> traits = []> :
        Op<JavaScript_Dialect, mnemonic, traits>;

#endif // JAVASCRIPT_DIALECT