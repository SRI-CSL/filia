#ifndef MLIR_DIALECT_JAVASCRIPT_JAVASCRIPT_TYPES_H_
#define MLIR_DIALECT_JAVASCRIPT_JAVASCRIPT_TYPES_H_

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/JavaScript/JavaScriptTypes.h.inc"



#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringMap.h"

#include "mlir/Dialect/Linalg/IR/LinalgOpsDialect.h.inc"

namespace mlir {
class MLIRContext;

namespace js {

/// A ValueType represents a JavaScript value.
class ValueType : public Type::TypeBase<ValueType, Type, TypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;
};

} // namespace js
} // namespace mlir

#endif // MLIR_DIALECT_JAVASCRIPT_JAVASCRIPT_TYPES_H_
