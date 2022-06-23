set(LLVM_TARGET_DEFINITIONS JavaScriptOps.td)
mlir_tablegen(JavaScriptOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(JavaScriptOpsEnums.cpp.inc -gen-enum-defs)

add_mlir_dialect(JavaScriptOps js)
add_mlir_doc(JavaScriptDialect JavaScriptDialect JavaScript/ -gen-dialect-doc)
add_mlir_doc(JavaScriptOps     JavaScriptOps     JavaScript/ -gen-op-doc)