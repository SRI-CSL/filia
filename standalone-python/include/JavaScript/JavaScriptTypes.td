// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MLIR_DIALECT_JAVASCRIPT_IR_JAVASCRIPTTYPES
#define MLIR_DIALECT_JAVASCRIPT_IR_JAVASCRIPTTYPES

include "JavaScript/JavaScriptDialect.td"

class JavaScript_Type<string name, string typeMnemonic>
    : TypeDef<JavaScript_Dialect, name, [], "::mlir::js::JavaScriptType"> {
  let mnemonic = typeMnemonic;
}

def JavaScript_Value : JavaScript_Type<"Value", "value"> {
  let summary = "JavaScript value";
  let description = [{
    This type represents a JavaScript value.
  }];
}

def JavaScript_Locals : JavaScript_Type<"Locals", "locals"> {
  let summary = "Mapping from variable names to the value";
  let description = [{
    This type represents a mapping from variable names to values.
  }];
}

#endif // MLIR_DIALECT_JAVASCRIPT_IR_JAVASCRIPTTYPES