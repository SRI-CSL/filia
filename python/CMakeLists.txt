include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_python`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_python.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(PythonDialectSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT PythonDialectSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_python"
  TD_FILE dialects/PythonOps.td
  SOURCES
    dialects/python.py
    dialects/_python_ops_ext.py
    _mlir_libs/_mlir/dialects/python.pyi

  DIALECT_NAME python)

declare_mlir_python_extension(PythonPythonSources.Extension
  MODULE_NAME _pythonDialects
  ADD_TO_PARENT PythonDialectSources
  SOURCES
    PythonExtension.cpp
  EMBED_CAPI_LINK_LIBS
    PythonDialectCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(PythonPythonCAPI
  INSTALL_COMPONENT PythonDialectModules
  INSTALL_DESTINATION python_packages/python/mlir_python/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/python/mlir_python/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    PythonDialectSources
    MLIRPythonSources.Core
)
target_compile_features(PythonPythonCAPI PRIVATE cxx_std_17)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(PythonDialectModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/python/mlir_python"
  INSTALL_PREFIX "python_packages/python/mlir_python"
  DECLARED_SOURCES
    PythonDialectSources
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    PythonPythonCAPI
  )

target_compile_features(PythonDialectModules.extension._mlir.dso PRIVATE cxx_std_17)
target_compile_features(PythonDialectModules.extension._mlirExecutionEngine.dso PRIVATE cxx_std_17)
target_compile_features(PythonDialectModules.extension._mlirDialectsLinalg.dso PRIVATE cxx_std_17)
target_compile_features(PythonDialectModules.extension._mlirDialectsPDL.dso PRIVATE cxx_std_17)
target_compile_features(PythonDialectModules.extension._mlirDialectsQuant.dso PRIVATE cxx_std_17)
target_compile_features(PythonDialectModules.extension._mlirDialectsSparseTensor.dso PRIVATE cxx_std_17)
target_compile_features(PythonDialectModules.extension._pythonDialects.dso PRIVATE cxx_std_17)