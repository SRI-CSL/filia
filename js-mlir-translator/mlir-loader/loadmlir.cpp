#include <stdio.h>
#include <mlir/Parser.h>
#include <mlir/InitAllDialects.h>

int main(int argc, const char** argv) {
  if (argc != 2) {
    fprintf(stderr, "Please provide the path to read.\n");
    return -1;
  }
  const char* path = argv[1];

  mlir::MLIRContext context;

  context.allowUnregisteredDialects();
  mlir::DialectRegistry registry;
  registry.insert<mlir::StandardOpsDialect, mlir::memref::MemRefDialect, mlir::omp::OpenMPDialect>();
  context.appendDialectRegistry(registry);
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