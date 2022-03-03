#include <stdio.h>
#include <mlir/Parser.h>
#include <mlir/InitAllDialects.h>

#include <mlir/Dialect/JavaScript/JavaScriptOps.h>
//#include "mlir/Dialect/JavaScript/JavaScriptOpsDialect.h.inc"

int main(int argc, const char** argv) {
  if (argc != 2) {
    fprintf(stderr, "Please provide the path to read.\n");
    return -1;
  }
  const char* path = argv[1];

  mlir::MLIRContext context;

//  context.allowUnregisteredDialects();
  mlir::DialectRegistry registry;
  printf("Starting\n");
  registry.insert<mlir::StandardOpsDialect, mlir::js::JavaScriptDialect>();
  context.appendDialectRegistry(registry);
  context.loadDialect<mlir::StandardOpsDialect>();
  context.loadDialect<mlir::js::JavaScriptDialect>();
  mlir::Block block;
  mlir::LogicalResult r = mlir::parseSourceFile(path, &block, &context);
  if (r.failed()) {
    fprintf(stderr, "Failed to parse mlir file file.\n");
    return -1;
  }
  for (auto i = block.begin(); i != block.end(); ++i) {
    printf("Name %s\n", i->getName().getStringRef().data());
  }

  return 0;
}