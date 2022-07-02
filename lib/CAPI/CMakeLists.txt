add_mlir_public_c_api_library(PythonDialectCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  MLIRPython
)
