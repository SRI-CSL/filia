get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithmetic
        MLIROptLib
        MLIRPython
        )
add_llvm_executable(python-opt python-opt.cpp PythonDomain.cpp)

llvm_update_compile_flags(python-opt)
target_link_libraries(python-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(python-opt)