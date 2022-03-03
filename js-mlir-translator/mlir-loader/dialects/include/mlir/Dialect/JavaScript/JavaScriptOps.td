include "mlir/Dialect/JavaScript/JavaScriptBase.td"

// Base class for JavaScript dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class JavaScript_Op<string mnemonic, list<OpTrait> traits = []> :
  Op<JavaScript_Dialect, mnemonic, traits>;

def UndefinedOp : JavaScript_Op<"undefined"> {
  let results = (outs ValueType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}