# Here we create a single aggregate shared library with the parts of the CAPI
# that we want to bundle together. Then we link a simple C executable
# against it to demonstrate that it does have the fully self contained
# core MLIR library and our own python dialect.
add_mlir_aggregate(StandaloneCAPITestLib
  SHARED
  EMBED_LIBS
  MLIRCAPIIR
  PythonDialectCAPI
)

add_llvm_executable(standalone-capi-test
  standalone-capi-test.c
)
llvm_update_compile_flags(standalone-capi-test)
target_link_libraries(standalone-capi-test
  PRIVATE StandaloneCAPITestLib)
target_compile_features(standalone-capi-test PRIVATE cxx_std_17)
