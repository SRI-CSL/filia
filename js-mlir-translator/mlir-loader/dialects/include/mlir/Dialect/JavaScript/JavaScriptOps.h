#ifndef MLIR_DIALECT_JAVASCRIPT_JAVASCRIPTOPS_H_
#define MLIR_DIALECT_JAVASCRIPT_JAVASCRIPTOPS_H_

//#include "mlir/IR/Attributes.h"
//#include "mlir/IR/Builders.h"
//#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
//#include "mlir/IR/Types.h"
//#include "mlir/Interfaces/SideEffectInterfaces.h"
//#include "llvm/Support/MathExtras.h"

#include "mlir/Dialect/JavaScript/JavaScriptDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/JavaScript/JavaScriptOps.h.inc"

#endif // MLIR_DIALECT_JAVASCRIPT_JAVASCRIPTOPS_H_