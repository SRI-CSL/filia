#include <stdio.h>
#include "mlir/IR/BuiltinDialect.h"
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
"  func @script_main() {\n"
"    %0 = python.undefined : !python.value\n"
"    %1 = python.truthy %0 : i1\n"
"    python.invoke %0(), ^bb1, ^bb2\n"
"  ^bb1(%a: !python.value):  // pred: ^bb0\n"
"    return\n"
"  ^bb2(%e: !python.value):  // pred: ^bb0\n"
"    %c = python.isinstance %e, \"StopIteration\" : i1\n"
"    cf.cond_br %c, ^bb3, ^bb4\n"
"  ^bb3:  // pred: ^bb2\n"
"    return\n"
"  ^bb4:  // pred: ^bb2\n"
"    python.throw %e\n"
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