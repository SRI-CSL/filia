// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef JAVASCRIPT_OPS
#define JAVASCRIPT_OPS

include "JavaScript/JavaScriptTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def JavaScript_NullOp : JavaScript_Op<"ull", [NoSideEffect]> {
    let summary = "Introduces a null JavaScript value.";
    let description = [{
        The `js.nulld` uses an operation trait to declare that it
        has no side effects.

        This operation returns a value.

        Example:

        ```mlir
        %0 = js.null
        ```
    }];

    let results = (outs JavaScript_Value:$result);
    let assemblyFormat = "attr-dict `:` type($result)";
}

def JavaScript_UndefinedOp : JavaScript_Op<"undefined", [NoSideEffect]> {
    let summary = "Introduces an undefined JavaScript value.";
    let description = [{
        The `js.undefined` uses an operation trait to declare that it
        has no side effects.

        This operation returns a value.

        Example:

        ```mlir
        %0 = js.undefined
        ```
    }];

    let results = (outs JavaScript_Value:$result);
    let assemblyFormat = "attr-dict `:` type($result)";
}

def JavaScript_EmptyLocals : JavaScript_Op<"empty_locals", [NoSideEffect]> {
    let summary = "Create an empty local table";
    let description = [{
        This creates an empty local table.
    }];
    let results = (outs JavaScript_Locals:$result);
    let assemblyFormat = "attr-dict `:` type($result)";
}

def JavaScript_Call : JavaScript_Op<"call", []> {
    let summary = "Call a function";
    let description = [{
        Operation that call a function.
    }];
    let arguments = (ins JavaScript_Value:$callee, OptionalAttr<UnitAttr>:$optional, OptionalAttr<ArrayAttr>:$spread, Variadic<JavaScript_Value>:$args);
    let results = (outs JavaScript_Value:$result);
    let assemblyFormat = "$callee `(` $args `)` attr-dict `:` type($result)";
}

#endif // JAVASCRIPT_OPS