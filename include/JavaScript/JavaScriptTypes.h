#ifndef MLIR_DIALECT_JAVASCRIPT_IR_JAVASCRIPTTYPES_H_
#define MLIR_DIALECT_JAVASCRIPT_IR_JAVASCRIPTTYPES_H_

#include "mlir/IR/Types.h"

//===----------------------------------------------------------------------===//
// JavaScriptDialect Types
//===----------------------------------------------------------------------===//

namespace mlir {
namespace js {
/// This class represents the base class of all JavaScript types.
class JavaScriptType : public Type {
public:
  using Type::Type;

  static bool classof(Type type);
};
} // namespace js
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "JavaScript/JavaScriptOpsTypes.h.inc"

#endif // MLIR_DIALECT_JAVASCRIPT_IR_JAVASCRIPTTYPES_H_