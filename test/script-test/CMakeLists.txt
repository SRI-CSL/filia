add_llvm_executable(script-test
  script-test.cpp
)
llvm_update_compile_flags(script-test)
target_link_libraries(script-test
  PRIVATE MLIRParser MLIRFuncDialect MLIRJavaScript MLIRPython)
mlir_check_all_link_libraries(script-test)
target_compile_features(script-test PRIVATE cxx_std_17)

