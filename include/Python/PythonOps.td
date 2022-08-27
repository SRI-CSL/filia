// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef PYTHON_OPS
#define PYTHON_OPS

include "Python/PythonTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

class Python_UnaryOp<string mnemonic> : Python_Op<mnemonic, []> {
    let arguments = (ins Python_Value: $arg);
    let results = (outs Python_ReturnValue:$result);
    let assemblyFormat = "$arg attr-dict `:` type($result)";
}

class Python_BinOp<string mnemonic> : Python_Op<mnemonic, []> {
    let arguments = (ins Python_Value: $lhs, Python_Value: $rhs);
    let results = (outs Python_ReturnValue:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// Unary operators
def Python_InvertOp : Python_UnaryOp<"invert">;
def Python_NotOp : Python_UnaryOp<"not">;
def Python_UAddOp : Python_UnaryOp<"uadd">;
def Python_USubOp : Python_UnaryOp<"usub">;

// Infix operators
def Python_AddOp      : Python_BinOp<"add">;
def Python_BitAndOp   : Python_BinOp<"bit_and">;
def Python_BitOrOp    : Python_BinOp<"bit_or">;
def Python_BitXorOp   : Python_BinOp<"bit_xor">;
def Python_DivOp      : Python_BinOp<"div">;
def Python_FloorDivOp : Python_BinOp<"floor_div">;
def Python_LShiftOp   : Python_BinOp<"lshift">;
def Python_ModOp      : Python_BinOp<"mod">;
def Python_MultOp     : Python_BinOp<"mult">;
def Python_MatMultOp  : Python_BinOp<"mat_mult">;
def Python_PowOp      : Python_BinOp<"pow">;
def Python_RShiftOp   : Python_BinOp<"rshift">;
def Python_SubOp      : Python_BinOp<"sub">;

// Comparison operators
def Python_EqOp    : Python_BinOp<"eq">;
def Python_GtOp    : Python_BinOp<"gt">;
def Python_GtEOp   : Python_BinOp<"gte">;
def Python_InOp    : Python_BinOp<"in">;
def Python_IsOp    : Python_BinOp<"is">;
def Python_IsNotOp : Python_BinOp<"is_not">;
def Python_LtOp    : Python_BinOp<"lt">;
def Python_LtEOp   : Python_BinOp<"lte">;
def Python_NotEqOp : Python_BinOp<"not_eq">;
def Python_NotInOp : Python_BinOp<"not_in">;

def Python_BuiltinAttr : I32EnumAttr<"Builtin", "", [
        I32EnumAttrCase<"abs", 0>,
        I32EnumAttrCase<"aiter", 1>,
        I32EnumAttrCase<"all", 2>,
        I32EnumAttrCase<"any", 3>,
        I32EnumAttrCase<"anext", 4>,
        I32EnumAttrCase<"ascii", 5>,
        I32EnumAttrCase<"bin", 6>,
        I32EnumAttrCase<"bool_builtin", 7>,
        I32EnumAttrCase<"breakpoint", 8>,
        I32EnumAttrCase<"bytearray", 9>,
        I32EnumAttrCase<"bytes", 10>,
        I32EnumAttrCase<"callable", 11>,
        I32EnumAttrCase<"chr", 12>,
        I32EnumAttrCase<"classmethod", 13>,
        I32EnumAttrCase<"compile", 14>,
        I32EnumAttrCase<"complex", 15>,
        I32EnumAttrCase<"delattr", 16>,
        I32EnumAttrCase<"dict", 17>,
        I32EnumAttrCase<"dir", 18>,
        I32EnumAttrCase<"divmod", 19>,
        I32EnumAttrCase<"enumerate", 20>,
        I32EnumAttrCase<"eval", 21>,
        I32EnumAttrCase<"exec", 22>,
        I32EnumAttrCase<"filter", 23>,
        I32EnumAttrCase<"float_builtin", 24>,
        I32EnumAttrCase<"format", 25>,
        I32EnumAttrCase<"frozenset", 26>,
        I32EnumAttrCase<"getattr", 27>,
        I32EnumAttrCase<"globals", 28>,
        I32EnumAttrCase<"hasattr", 29>,
        I32EnumAttrCase<"hash", 30>,
        I32EnumAttrCase<"help", 31>,
        I32EnumAttrCase<"hex", 32>,
        I32EnumAttrCase<"id", 33>,
        I32EnumAttrCase<"import", 34>,
        I32EnumAttrCase<"input", 35>,
        I32EnumAttrCase<"int_builtin", 36>,
        I32EnumAttrCase<"isinstance", 37>,
        I32EnumAttrCase<"issubclass", 38>,
        I32EnumAttrCase<"iter", 39>,
        I32EnumAttrCase<"len", 40>,
        I32EnumAttrCase<"list", 41>,
        I32EnumAttrCase<"locals", 42>,
        I32EnumAttrCase<"map", 43>,
        I32EnumAttrCase<"max", 44>,
        I32EnumAttrCase<"memoryview", 45>,
        I32EnumAttrCase<"min", 46>,
        I32EnumAttrCase<"next", 47>,
        I32EnumAttrCase<"object", 48>,
        I32EnumAttrCase<"oct", 49>,
        I32EnumAttrCase<"open", 50>,
        I32EnumAttrCase<"ord", 51>,
        I32EnumAttrCase<"pow", 52>,
        I32EnumAttrCase<"print", 53>,
        I32EnumAttrCase<"property", 54>,
        I32EnumAttrCase<"range", 55>,
        I32EnumAttrCase<"repr", 56>,
        I32EnumAttrCase<"reversed", 57>,
        I32EnumAttrCase<"round", 58>,
        I32EnumAttrCase<"scriptmain", 59>,
        I32EnumAttrCase<"set", 60>,
        I32EnumAttrCase<"setattr", 61>,
        I32EnumAttrCase<"slice", 62>,
        I32EnumAttrCase<"sorted", 63>,
        I32EnumAttrCase<"staticmethod", 64>,
        I32EnumAttrCase<"str", 65>,
        I32EnumAttrCase<"sum", 66>,
        I32EnumAttrCase<"super", 67>,
        I32EnumAttrCase<"tuple", 68>,
        I32EnumAttrCase<"type", 69>,
        I32EnumAttrCase<"vars", 70>,
        I32EnumAttrCase<"zip", 71>
    ]> {
  let cppNamespace = "::mlir::python";
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

def Python_ArraySetOp : Python_Op<"array_set", []> {
    let summary = "Update array at a given index.";
    let description = [{
        FIXME
    }];

    let arguments = (ins Python_Value:$array, Python_Value:$index, Python_Value: $value);
    let assemblyFormat = "$array $index $value attr-dict";
}

def Python_BuiltinOp : Python_Op<"builtin", [NoSideEffect]> {
    let summary = "A reference to a builtin function.";
    let description = [{
        FIXME
    }];

    let arguments = (ins Python_BuiltinAttr:$val);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$val attr-dict `:` type($result)";
}

def Python_ModuleOp : Python_Op<"module", [NoSideEffect]> {
    let summary = "A reference to an external module.";
    let description = [{
        FIXME
    }];

    let arguments = (ins StrAttr:$val);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$val attr-dict `:` type($result)";
}

def Python_S64LitOp : Python_Op<"intlit", [NoSideEffect]> {
    let summary = "An integer literal.";
    let description = [{
        FIXME
    }];

    let arguments = (ins SI64Attr:$val);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$val attr-dict `:` type($result)";
}

def Python_IntLitOp : Python_Op<"u64lit", [NoSideEffect]> {
    let summary = "An integer literal.";
    let description = [{
        FIXME
    }];

    let arguments = (ins StrAttr:$val);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$val attr-dict `:` type($result)";
}

def Python_StrLitOp : Python_Op<"strlit", [NoSideEffect]> {
    let summary = "A string literal.";
    let description = [{
        FIXME
    }];

    let arguments = (ins StrAttr:$val);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$val attr-dict `:` type($result)";
}

def Python_GetMethodOp : Python_Op<"getmethod", []> {
    let summary = "Returns a method from a value..";
    let description = [{
        FIXME
    }];

    let arguments = (ins Python_Value:$obj, StrAttr:$method);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$obj `,` $method attr-dict `:` type($result)";
}

def Python_InvokeOp : Python_Op<"invoke", []> {
    let summary = "Invoke a Python method";
    let description = [{
        Operation that invokes a Python method
    }];
    let arguments = (ins Python_Value:$callee,
                     Variadic<Python_Value>:$args,
                     OptionalAttr<ArrayAttr>:$keywords
                     );
    let results = (outs Python_ReturnValue:$result);
    let assemblyFormat = "$callee `(` $args `)` attr-dict `:` type($result)";
}

def Python_IsInstanceOp : Python_Op<"isinstance", [NoSideEffect]> {
    let summary = "Test if value matches an instance.";
    let description = [{
        Operation that test if a value is an instance.
    }];
    let arguments = (ins Python_Value:$value, StrAttr:$classname);
    let results = (outs I1:$result);
    let assemblyFormat = "$value `,` $classname attr-dict `:` type($result)";
}

def Python_FormattedString : Python_Op<"formattedstring", [NoSideEffect]> {
    let summary = "An interpolated string";
    let description = [{
        Operation that evaluates an interpolated string (see PEP 498).
    }];
    let arguments = (ins Variadic<Python_Value>:$args);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "`(` $args `)` attr-dict `:` type($result)";
}

def Python_ListOp : Python_Op<"list", [NoSideEffect]> {
    let summary = "Construct a list.";
    let description = [{
        Constructs a list from a list of arguments.
    }];
    let arguments = (ins Variadic<Python_Value>:$args);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "`(` $args `)` attr-dict `:` type($result)";
}

def Python_TupleOp : Python_Op<"tuple", [NoSideEffect]> {
    let summary = "Construct a tuple.";
    let description = [{
        Constructs a tuple from a list of arguments.
    }];
    let arguments = (ins Variadic<Python_Value>:$args);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "`(` $args `)` attr-dict `:` type($result)";
}

def Python_TruthyOp : Python_Op<"truthy", [NoSideEffect]> {
    let summary = "Predicate that checks if a value is truthy.";
    let description = [{
        Checks if value is one considered truthy in Python
    }];
    let arguments = (ins Python_Value:$val);
    let results = (outs I1:$result);
    let assemblyFormat = "$val attr-dict `:` type($result)";
}

def Python_TupleCheckOp : Python_Op<"tuple_check", []> {
    let summary = "Predicate that checks if a value is a tuple.";
    let description = [{
        Checks if value is a tuple with the given arity.
    }];
    let arguments = (ins Python_Value:$value, SI64Attr:$arity);
    let assemblyFormat = "$value `,` $arity attr-dict";
}

def Python_TupleGetOp : Python_Op<"tuple_get", [NoSideEffect]> {
    let summary = "Predicate that gets a value with given index from tuple.";
    let description = [{
        Get a value from tuple (undefined if value is not a tuple or index is invalid)
    }];
    let arguments = (ins Python_Value:$value, SI64Attr:$idx);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$value `,` $idx attr-dict `:` type($result)";
}

def Python_TypeOp : Python_Op<"type", [NoSideEffect]> {
    let summary = "Returns type of a given value.";
    let description = [{
        Returns type of a value in Python.
    }];
    let arguments = (ins Python_Value:$value);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$value attr-dict `:` type($result)";
}

def Python_FunctionRefOp : Python_Op<"function_ref", [NoSideEffect]> {
    let summary = "Create a Python value referencing a function.";
    let arguments =
        (ins SymbolRefAttr:$fun,
             ArrayAttr:$arg_names,
             Python_Scope:$scope,
             Variadic<Python_Cell>:$capture);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$fun `(` $arg_names `)` $scope `[` $capture `]` attr-dict `:` type($result)";
}

// Return  operators

def Python_MkReturnOp : Python_Op<"mk_return", [NoSideEffect]> {
    let summary = "Create a normal return value.";
    let description = [{
        This takes a Python value and constructs a return value.
        Stores the value in the cell.
    }];
    let arguments = (ins Python_Value:$value);
    let results = (outs Python_ReturnValue:$result);
    let assemblyFormat = "$value attr-dict `:` type($result)";
}

def Python_MkExceptOp : Python_Op<"mk_except", [NoSideEffect]> {
    let summary = "Create an exception return value.";
    let description = [{
        Stores the value in the cell.
    }];
    let arguments = (ins Python_Value:$value, Python_Value:$traceback);
    let results = (outs Python_ReturnValue:$result);
    let assemblyFormat = "$value `,` $traceback attr-dict `:` type($result)";
}

def Python_RetBranchOp : Python_Op<"ret_branch", [
        AttrSizedOperandSegments,
        DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
        NoSideEffect,
        Terminator]> {
    let summary = "Jump to one block or another due to a return.";
    let arguments = (ins Python_ReturnValue:$value,
                     Variadic<AnyType>:$returnDestOperands,
                     Variadic<AnyType>:$exceptDestOperands
                     );
    let successors = (successor AnySuccessor:$returnDest, AnySuccessor:$exceptDest);
    let assemblyFormat = [{
      $value `,`
      $returnDest (`(` $returnDestOperands^ `:` type($returnDestOperands) `)`)? `,`
      $exceptDest (`(` $exceptDestOperands^ `:` type($exceptDestOperands) `)`)?
      attr-dict
    }];
  let extraClassDeclaration = [{
    // These are the indices into the dests list.
    enum { returnIndex = 0, exceptIndex = 1 };
  }];
}

// Cell operators

def Python_CellAllocOp : Python_Op<"cell_alloc", [NoSideEffect]> {
    let summary = "Create a cell for storing values.";
    let description = [{
        This creates a cell with an optional initial value.
        If no value is defined then the cell is empty.
    }];
    let arguments = (ins Optional<Python_Value>:$initial);
    let results = (outs Python_Cell:$result);
    let assemblyFormat = "$initial attr-dict `:` type($result)";
}

def Python_CellStoreOp : Python_Op<"cell_store", []> {
    let summary = "Stores the value in the cell.";
    let description = [{
        Stores the value in the cell.
    }];
    let arguments = (ins Python_Cell:$cell, Python_Value:$value);
    let assemblyFormat = "$cell `,` $value attr-dict";
}

def Python_CellLoadOp : Python_Op<"cell_load", []> {
    let summary = "Returns the value currently stored in the cell.";
    let description = [{
        Returns the value currently stored in the cell.
        This uses a ReturnValue so load returns the value if it is assigned
        and an exception of type UnboundLocalError if the cell is empty.
    }];
    let arguments = (ins Python_Cell:$cell, StrAttr:$name);
    let results = (outs Python_ReturnValue:$result);
    let assemblyFormat = "$cell attr-dict `:` type($result)";
}

// Scope operators

def Python_ScopeInitOp : Python_Op<"scope_init", [NoSideEffect]> {
    let summary = "Create an empty scope.";
    let description = [{
        This creates an initial empty scope.
    }];
    let arguments = (ins ArrayAttr:$names, Variadic<Python_Cell>:$cells);
    let results = (outs Python_Scope:$result);
    let assemblyFormat = "$names `(` $cells `)` attr-dict `:` type($result)";
}

def Python_ScopeExtendOp : Python_Op<"scope_extend", [NoSideEffect]> {
    let summary = "Create a scope that extends an existing one";
    let description = [{
        This creates an empty scope that extends a current one.
    }];
    let arguments = (ins Python_Scope:$scope, ArrayAttr:$names, Variadic<Python_Cell>:$cells);
    let results = (outs Python_Scope:$result);
    let assemblyFormat = "$scope $names `(` $cells `)` attr-dict `:` type($result)";
}

def Python_ScopeGetOp : Python_Op<"scope_get", []> {
    let summary = "Get value of a variable in local/global map.";
    let description = [{
        This gets a variable in the map.
    }];
    let arguments = (ins Python_Scope:$scope, StrAttr:$name);
    let results = (outs Python_Cell:$result);
    let assemblyFormat = "$scope `,` $name attr-dict `:` type($result)";
}

#endif // PYTHON_OPS
