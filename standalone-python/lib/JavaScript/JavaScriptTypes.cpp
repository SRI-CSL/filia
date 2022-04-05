// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "JavaScript/JavaScriptDialect.h"
#include "JavaScript/JavaScriptTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::js;

//===----------------------------------------------------------------------===//
// TableGen'd type method definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "JavaScript/JavaScriptOpsTypes.cpp.inc"

bool JavaScriptType::classof(Type type) {
  return llvm::isa<JavaScriptDialect>(type.getDialect());
}