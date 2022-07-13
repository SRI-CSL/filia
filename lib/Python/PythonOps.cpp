// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "Python/PythonOps.h"
#include "Python/PythonDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "Python/PythonOps.cpp.inc"

#include "Python/PythonOpsEnums.cpp.inc"

namespace mlir {
namespace python {

Block* RetBranchOp::getSuccessorForOperands(ArrayRef<Attribute> operands) {
  return nullptr;
}

/*
Optional<MutableOperandRange> RetBranchOp::getMutableSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return index == returnIndex ? returnDestOperandsMutable()
                              : exceptDestOperandsMutable();
}
*/

SuccessorOperands RetBranchOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return index == returnIndex
          ? SuccessorOperands(1, returnDestOperandsMutable())
          : SuccessorOperands(2, exceptDestOperandsMutable());
}

}
}