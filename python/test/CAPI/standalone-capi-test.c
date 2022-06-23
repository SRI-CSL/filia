//===- standalone-cap-demo.c - Simple demo of C-API -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: standalone-capi-test 2>&1 | FileCheck %s

#include <stdio.h>

#include "mlir-c/IR.h"
#include "Python-c/Dialects.h"

int main(int argc, char **argv) {
  MlirContext ctx = mlirContextCreate();
  // TODO: Create the dialect handles for the builtin dialects and avoid this.
  // This adds dozens of MB of binary size over just the standalone dialect.
  mlirRegisterAllDialects(ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__python__(), ctx);

  MlirModule module = mlirModuleCreateParse(ctx,
    mlirStringRefCreateFromCString(
      "%0 = python.undefined : !python.value\n"));
  if (mlirModuleIsNull(module)) {
    printf("ERROR: Could not parse.\n");
    mlirContextDestroy(ctx);
    return 1;
  }
  MlirOperation op = mlirModuleGetOperation(module);

  // CHECK: python.undefined : !python.value
  mlirOperationDump(op);

  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
  return 0;
}
