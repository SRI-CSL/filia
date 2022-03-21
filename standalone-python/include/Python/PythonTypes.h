#ifndef MLIR_DIALECT_PYTHON_IR_PYTHONTYPES_H_
#define MLIR_DIALECT_PYTHON_IR_PYTHONTYPES_H_

#include "mlir/IR/Types.h"

//===----------------------------------------------------------------------===//
// Python Dialect Types
//===----------------------------------------------------------------------===//

namespace mlir {
namespace python {
/// This class represents the base class of all PDL types.
class PythonType : public Type {
public:
  using Type::Type;

  static bool classof(Type type);
};
} // namespace python
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "Python/PythonOpsTypes.h.inc"

#endif // MLIR_DIALECT_PYTHON_IR_PYTHONTYPES_H_