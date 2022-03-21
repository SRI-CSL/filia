// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Python/PythonDialect.h"
#include "Python/PythonOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::python;

#include "Python/PythonOpsDialect.cpp.inc"

void PythonDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Python/PythonOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Python/PythonOpsTypes.cpp.inc"
      >();
}
