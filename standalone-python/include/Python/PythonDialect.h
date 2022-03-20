// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef PYTHON_PYTHONDIALECT_H
#define PYTHON_PYTHONDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

#include "Python/PythonOpsDialect.h.inc"

namespace mlir {
namespace python {

/// A ValueType represents a JavaScript value.
class ValueType : public Type::TypeBase<ValueType, Type, TypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;
};

} // namespace python
} // namespace mlir

#endif // PYTHON_PYTHONDIALECT_H
