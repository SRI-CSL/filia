get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithmetic
        MLIROptLib
        MLIRPython
        )
add_llvm_executable(script-opt script-opt.cpp)

llvm_update_compile_flags(script-opt)
target_link_libraries(script-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(script-opt)
