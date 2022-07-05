// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "JavaScript/JavaScriptDialect.h"
#include "Python/PythonDialect.h"
#include "Python/PythonLoadStorePass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  // Register standalone passes here.
  mlir::python::registerLoadStorePass();

  mlir::DialectRegistry registry;
  registry.insert<mlir::js::JavaScriptDialect,
                  mlir::cf::ControlFlowDialect,
                  mlir::python::PythonDialect,
                  mlir::arith::ArithmeticDialect,
                  mlir::func::FuncDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Python optimizer driver\n", registry));
}
