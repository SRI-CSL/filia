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

class Python_UnaryOp<string mnemonic> :
        Python_Op<mnemonic, [AttrSizedOperandSegments,Terminator]> {
    let arguments = (ins Python_Value: $arg,
                         Variadic<AnyType>:$returnDestOperands,
                         Variadic<AnyType>:$exceptDestOperands
                     );
    let successors = (successor AnySuccessor:$returnDest, AnySuccessor:$exceptDest);
    let assemblyFormat = [{
      $arg `,`
      $returnDest (`(` $returnDestOperands^ `:` type($returnDestOperands) `)`)? `,`
      $exceptDest (`(` $exceptDestOperands^ `:` type($exceptDestOperands) `)`)?
      attr-dict
    }];
}

class Python_BinOp<string mnemonic> :
        Python_Op<mnemonic, [AttrSizedOperandSegments,Terminator]> {
    let arguments = (ins Python_Value: $lhs,
                         Python_Value: $rhs,
                         Variadic<AnyType>:$returnDestOperands,
                         Variadic<AnyType>:$exceptDestOperands
                     );
    let successors = (successor AnySuccessor:$returnDest, AnySuccessor:$exceptDest);
    let assemblyFormat = [{
      $lhs `,` $rhs `,`
      $returnDest (`(` $returnDestOperands^ `:` type($returnDestOperands) `)`)? `,`
      $exceptDest (`(` $exceptDestOperands^ `:` type($exceptDestOperands) `)`)?
      attr-dict
    }];
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

def Python_BuiltinAttr : StrEnumAttr<"BuiltinAttr", "", [
        StrEnumAttrCase<"abs">,
        StrEnumAttrCase<"aiter">,
        StrEnumAttrCase<"all">,
        StrEnumAttrCase<"any">,
        StrEnumAttrCase<"anext">,
        StrEnumAttrCase<"ascii">,
        StrEnumAttrCase<"bin">,
        StrEnumAttrCase<"bool_builtin">,
        StrEnumAttrCase<"breakpoint">,
        StrEnumAttrCase<"bytearray">,
        StrEnumAttrCase<"bytes">,
        StrEnumAttrCase<"callable">,
        StrEnumAttrCase<"chr">,
        StrEnumAttrCase<"classmethod">,
        StrEnumAttrCase<"compile">,
        StrEnumAttrCase<"complex">,
        StrEnumAttrCase<"delattr">,
        StrEnumAttrCase<"dict">,
        StrEnumAttrCase<"dir">,
        StrEnumAttrCase<"divmod">,
        StrEnumAttrCase<"enumerate">,
        StrEnumAttrCase<"eval">,
        StrEnumAttrCase<"exec">,
        StrEnumAttrCase<"filter">,
        StrEnumAttrCase<"float_builtin">,
        StrEnumAttrCase<"format">,
        StrEnumAttrCase<"frozenset">,
        StrEnumAttrCase<"getattr">,
        StrEnumAttrCase<"globals">,
        StrEnumAttrCase<"hasattr">,
        StrEnumAttrCase<"hash">,
        StrEnumAttrCase<"help">,
        StrEnumAttrCase<"hex">,
        StrEnumAttrCase<"id">,
        StrEnumAttrCase<"input">,
        StrEnumAttrCase<"int_builtin">,
        StrEnumAttrCase<"isinstance">,
        StrEnumAttrCase<"issubclass">,
        StrEnumAttrCase<"iter">,
        StrEnumAttrCase<"len">,
        StrEnumAttrCase<"list">,
        StrEnumAttrCase<"locals">,
        StrEnumAttrCase<"map">,
        StrEnumAttrCase<"max">,
        StrEnumAttrCase<"memoryview">,
        StrEnumAttrCase<"min">,
        StrEnumAttrCase<"next">,
        StrEnumAttrCase<"object">,
        StrEnumAttrCase<"oct">,
        StrEnumAttrCase<"open">,
        StrEnumAttrCase<"ord">,
        StrEnumAttrCase<"pow">,
        StrEnumAttrCase<"print">,
        StrEnumAttrCase<"property">,
        StrEnumAttrCase<"range">,
        StrEnumAttrCase<"repr">,
        StrEnumAttrCase<"reversed">,
        StrEnumAttrCase<"round">,
        StrEnumAttrCase<"set">,
        StrEnumAttrCase<"setattr">,
        StrEnumAttrCase<"slice">,
        StrEnumAttrCase<"sorted">,
        StrEnumAttrCase<"staticmethod">,
        StrEnumAttrCase<"str">,
        StrEnumAttrCase<"sum">,
        StrEnumAttrCase<"super">,
        StrEnumAttrCase<"tuple">,
        StrEnumAttrCase<"type">,
        StrEnumAttrCase<"vars">,
        StrEnumAttrCase<"zip">,
        StrEnumAttrCase<"import">
    ]> {
    let cppNamespace = "::mlir::python";
}

def Python_ScopeInit : Python_Op<"scope_init", [NoSideEffect]> {
    let summary = "Create an empty scope.";
    let description = [{
        This creates an initial empty scope.
    }];
    let results = (outs Python_Scope:$result);
    let assemblyFormat = "attr-dict `:` type($result)";
}

def Python_ScopeExtend : Python_Op<"scope_extend", [NoSideEffect]> {
    let summary = "Create a scope that extends an existing one";
    let description = [{
        This creates an empty scope that extends a current one.
    }];
    let arguments = (ins Python_Scope:$scope);
    let results = (outs Python_Scope:$result);
    let assemblyFormat = "$scope attr-dict `:` type($result)";
}

def Python_ScopeImport : Python_Op<"scope_import", []> {
    let summary = "Import a module into the scope.";
    let description = [{
        This binds a variable in a scope table.
    }];
    let arguments = (ins Python_Scope:$scope, StrAttr:$module, StrAttr:$asName);
    let assemblyFormat = "$scope `,` $module `,` $asName attr-dict";
}

def Python_ScopeSet : Python_Op<"scope_set", []> {
    let summary = "Define a variable in a scope.";
    let description = [{
        This binds a variable in a scope table.
    }];
    let arguments = (ins Python_Scope:$scope, StrAttr:$name, Python_Value:$value);
    let assemblyFormat = "$scope `,` $name `,` $value attr-dict";
}

def Python_ScopeGet : Python_Op<"scope_get", []> {
    let summary = "Get value of a variable in local/global map.";
    let description = [{
        This gets a variable in the map.
    }];
    let arguments = (ins Python_Scope:$scope, StrAttr:$name);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$scope `,` $name attr-dict `:` type($result)";
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

def Python_Builtin : Python_Op<"builtin", [NoSideEffect]> {
    let summary = "A reference to a builtin function.";
    let description = [{
        FIXME
    }];

    let arguments = (ins Python_BuiltinAttr:$val);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$val attr-dict `:` type($result)";
}

def Python_Module : Python_Op<"module", [NoSideEffect]> {
    let summary = "A reference to an external module.";
    let description = [{
        FIXME
    }];

    let arguments = (ins StrAttr:$val);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$val attr-dict `:` type($result)";
}

def Python_S64Lit : Python_Op<"intlit", [NoSideEffect]> {
    let summary = "An integer literal.";
    let description = [{
        FIXME
    }];

    let arguments = (ins SI64Attr:$val);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$val attr-dict `:` type($result)";
}

def Python_IntLit : Python_Op<"u64lit", [NoSideEffect]> {
    let summary = "An integer literal.";
    let description = [{
        FIXME
    }];

    let arguments = (ins StrAttr:$val);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$val attr-dict `:` type($result)";
}

def Python_StrLit : Python_Op<"strlit", [NoSideEffect]> {
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

def Python_InvokeOp : Python_Op<"invoke", [
      AttrSizedOperandSegments,Terminator
    ]> {
    let summary = "Invoke a Python method";
    let description = [{
        Operation that invokes a Python method
    }];
    let arguments = (ins Python_Value:$callee,
                     Variadic<Python_Value>:$args,
                     OptionalAttr<ArrayAttr>:$keywords,
                     Variadic<AnyType>:$returnDestOperands,
                     Variadic<AnyType>:$exceptDestOperands
                     );

    let successors = (successor AnySuccessor:$returnDest, AnySuccessor:$exceptDest);
    let assemblyFormat = [{
      $callee `(` $args `)` `,`
      $returnDest (`(` $returnDestOperands^ `:` type($returnDestOperands) `)`)? `,`
      $exceptDest (`(` $exceptDestOperands^ `:` type($exceptDestOperands) `)`)?
      attr-dict
    }];
}

def Python_ThrowOp : Python_Op<"throw", [Terminator]> {
    let summary = "Throw a Python exception.";
    let description = [{
        Operation that invokes a Python method
    }];
    let arguments = (ins Python_Value:$except);
    let assemblyFormat = "$except attr-dict";
}

def Python_IsInstance : Python_Op<"isinstance", [NoSideEffect]> {
    let summary = "Test if value matches an instance.";
    let description = [{
        Operation that test if a value is an instance.
    }];
    let arguments = (ins Python_Value:$value, StrAttr:$classname);
    let results = (outs I1:$result);
    let assemblyFormat = "$value `,` $classname attr-dict `:` type($result)";
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

def Python_List : Python_Op<"list", []> {
    let summary = "Construct a list.";
    let description = [{
        Constructs a list from a list of arguments.
    }];
    let arguments = (ins Variadic<Python_Value>:$args);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "`(` $args `)` attr-dict `:` type($result)";
}

def Python_Tuple : Python_Op<"tuple", []> {
    let summary = "Construct a tuple.";
    let description = [{
        Constructs a tuple from a list of arguments.
    }];
    let arguments = (ins Variadic<Python_Value>:$args);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "`(` $args `)` attr-dict `:` type($result)";
}

def Python_Truthy : Python_Op<"truthy", [NoSideEffect]> {
    let summary = "Predicate that checks if a value is truthy.";
    let description = [{
        Checks if value is one considered truthy in Python
    }];
    let arguments = (ins Python_Value:$val);
    let results = (outs I1:$result);
    let assemblyFormat = "$val attr-dict `:` type($result)";
}

def Python_TupleCheck : Python_Op<"tuple_check", []> {
    let summary = "Predicate that checks if a value is a tuple.";
    let description = [{
        Checks if value is a tuple with the given arity.
    }];
    let arguments = (ins Python_Value:$value, SI64Attr:$arity);
    let assemblyFormat = "$value `,` $arity attr-dict";
}

def Python_TupleGet : Python_Op<"tuple_get", []> {
    let summary = "Predicate that gets a value with given index from tuple.";
    let description = [{
        Get a value from tuple (undefined if value is not a tuple or index is invalid)
    }];
    let arguments = (ins Python_Value:$value, SI64Attr:$idx);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$value `,` $idx attr-dict `:` type($result)";
}

def Python_FunctionRef : Python_Op<"function_ref", []> {
    let summary = "Create a Python value referencing a function.";
    let arguments = (ins SymbolRefAttr:$fun, ArrayAttr:$arg_names, Python_Scope:$scope);
    let results = (outs Python_Value:$result);
    let assemblyFormat = "$fun `(` $arg_names `)` $scope attr-dict `:` type($result)";
}

#endif // PYTHON_OPS
