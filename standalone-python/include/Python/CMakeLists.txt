set(LLVM_TARGET_DEFINITIONS PythonOps.td )
mlir_tablegen(PythonOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(PythonOpsEnums.cpp.inc -gen-enum-defs)

add_mlir_dialect(PythonOps python)
add_mlir_doc(PythonDialect PythonDialect Python/ -gen-dialect-doc)
add_mlir_doc(PythonOps PythonOps Python/ -gen-op-doc)
