#include <stdio.h>
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"

#include "JavaScript/JavaScriptDialect.h"
#include "Python/PythonDialect.h"


void testJavascript() {
  mlir::MLIRContext context;
  context.loadDialect<mlir::js::JavaScriptDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> owning =
      mlir::parseSourceString<mlir::ModuleOp>(
        "%0 = js.undefined : !js.value\n"
        "%1 = js.empty_locals : !js.locals\n"
        "%2 = js.call %0(%0) {js.optional, js.spread=[0]} : !js.value\n"
        , &context);

  if (!owning) {
    printf("ERROR: Could not parse.\n");
    exit(-1);
  }
}


void testPython() {
  mlir::MLIRContext context;
  context.loadDialect<mlir::cf::ControlFlowDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::python::PythonDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> owning =
      mlir::parseSourceString<mlir::ModuleOp>(
"module {\n"
"  func.func @script_main() -> !python.return_value {\n"
"    %0 = python.undefined : !python.value\n"
"    %1 = python.truthy %0 : i1\n"
"    %2 = python.invoke %0() : !python.return_value\n"
"    python.ret_branch %2, ^bb1, ^bb2\n"
"  ^bb1(%a: !python.value):  // pred: ^bb0\n"
"    %3 = python.mk_return %a : !python.return_value\n"
"    return %3 : !python.return_value\n"
"  ^bb2(%e: !python.value, %tb : !python.value):  // pred: ^bb0\n"
"    %c = python.isinstance %e, \"StopIteration\" : i1\n"
"    cf.cond_br %c, ^bb3, ^bb4\n"
"  ^bb3:  // pred: ^bb2\n"
"    %4 = python.mk_return %0 : !python.return_value\n"
"    return %4 : !python.return_value\n"
"  ^bb4:  // pred: ^bb2\n"
"    %5 = python.mk_except %e, %tb: !python.return_value\n"
"    return %5  : !python.return_value\n"
"  }\n"
"}\n"
        , &context);
  if (!owning) {
    printf("ERROR: Could not parse.\n");
    exit(-1);
  }
  owning->dump();
}

int main(int argc, char** argv) {
  testJavascript();
  testPython();
  return 0;
}