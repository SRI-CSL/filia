// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef PYTHON_OPS
#define PYTHON_OPS

include "Python/PythonTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"


def Python_BuiltinGetAttr: StrEnumAttrCase<"getattr">;
def Python_BuiltinAttr: StrEnumAttr<"BuiltinAttr", "", [Python_BuiltinGetAttr]>;

def Python_EmptyLocals : Python_Op<"empty_locals", [NoSideEffect]> {
    let summary = "Create an empty local table";
    let description = [{
        This creates an empty local table.
    }];
    let results = (outs Python_Locals:$result);
    let assemblyFormat = "attr-dict `:` type($result)";
}

def Python_LocalSet : Python_Op<"local_set", [NoSideEffect]> {
    let summary = "Define a variable in a local map.";
    let description = [{
        This binds a variable in a local table.
    }];
    let arguments = (ins Python_Locals:$map, StrAttr:$name, Python_Value:$value);
    let results = (outs Python_Locals:$result);
    let assemblyFormat = "$map `,` $name `,` $value attr-dict `:` type($result)";
}

def Python_Get : Python_Op<"get", [NoSideEffect]> {
    let summary = "Get value of a variable in local/global map.";
    let description = [{
        This gets a variable in the map.
    }];
    let arguments = (ins Python_Locals:$map, StrAttr:$name);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$map `,` $name attr-dict `:` type($result)";
}


def Python_NoneOp : Python_Op<"none", [NoSideEffect]> {
    let summary = "Introduces the None Python value.";
    let description = [{
        The `python.none` returns the "None" value.
    }];

    let results = (outs Python_Value:$result);
    let assemblyFormat = "attr-dict `:` type($result)";
}


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

    let results = (outs Python_Value:$result);
    let assemblyFormat = "attr-dict `:` type($result)";
}

def Python_IntLit : Python_Op<"intlit", []> {
    let summary = "An integer literal.";
    let description = [{
        FIXME
    }];

    let arguments = (ins SI64Attr:$val);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$val attr-dict `:` type($result)";
}

def Python_Builtin : Python_Op<"builtin", []> {
    let summary = "A reference to a builtin function.";
    let description = [{
        FIXME
    }];

    let arguments = (ins Python_BuiltinAttr:$val);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$val attr-dict `:` type($result)";
}

def Python_StrLit : Python_Op<"strlit", []> {
    let summary = "A string literal.";
    let description = [{
        FIXME
    }];

    let arguments = (ins StrAttr:$val);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$val attr-dict `:` type($result)";
}

def Python_GetMethod : Python_Op<"getmethod", []> {
    let summary = "Returns a method from a value..";
    let description = [{
        FIXME
    }];

    let arguments = (ins Python_Value:$obj, StrAttr:$method);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$obj `,` $method attr-dict `:` type($result)";
}

def Python_Invoke : Python_Op<"invoke", []> {
    let summary = "Invoke a Python method";
    let description = [{
        Operation that invokes a Python method
    }];
    let arguments = (ins Python_Value:$callee, Variadic<Python_Value>:$args);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$callee `(` $args `)` attr-dict `:` type($result)";
}

def Python_FormattedString : Python_Op<"formattedstring", []> {
    let summary = "An interpolated string";
    let description = [{
        Operation that evaluates an interpolated string (see PEP 498).
    }];
    let arguments = (ins Variadic<Python_Value>:$args);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "`(` $args `)` attr-dict `:` type($result)";
}

#endif // PYTHON_OPS
