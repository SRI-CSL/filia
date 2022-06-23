// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "JavaScript/JavaScriptDialect.h"
#include "JavaScript/JavaScriptOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::js;

#include "JavaScript/JavaScriptOpsDialect.cpp.inc"

void JavaScriptDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "JavaScript/JavaScriptOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "JavaScript/JavaScriptOpsTypes.cpp.inc"
      >();
}