// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Python/PythonDialect.h"
#include "Python/PythonOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::python;

#include "Python/PythonOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Python dialect.
//===----------------------------------------------------------------------===//

static void print(ValueType rt, DialectAsmPrinter &os) { os << "value"; }

void PythonDialect::printType(Type type, DialectAsmPrinter &os) const {
  print(type.cast<ValueType>(), os);
}

/// Parse a type registered to this dialect.
Type PythonDialect::parseType(::mlir::DialectAsmParser &parser) const {
  // Parse the main keyword for the type.
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  MLIRContext *context = getContext();

  // Handle 'range' types.
  if (keyword == "value")
    return ValueType::get(context);

  parser.emitError(parser.getNameLoc(), "unknown Python dialect type: " + keyword);
  return Type();
}

void PythonDialect::initialize() {
  addTypes<ValueType>();
  addOperations<
#define GET_OP_LIST
#include "Python/PythonOps.cpp.inc"
      >();
}
