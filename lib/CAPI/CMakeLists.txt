add_mlir_public_c_api_library(PythonDialectCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  MLIRPython
)
target_compile_features(obj.PythonDialectCAPI PRIVATE cxx_std_17)
