#include <stdio.h>
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser.h"

#include "JavaScript/JavaScriptDialect.h"
#include "Python/PythonDialect.h"

int main(int argc, char** argv) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::js::JavaScriptDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> owning =
      mlir::parseSourceString(
        "%0 = js.undefined : !js.value\n"
        "%1 = js.empty_locals : !js.locals\n"
        "%2 = js.call %0(%0) {js.optional, js.spread=[0]} : !js.value\n"
        , &context);

  if (!owning) {
    printf("ERROR: Could not parse.\n");
    return 1;
  }
  owning->dump();
  return 0;
}