# Build directory
set(CONFIG_PY_DIR "${MLIR_BINARY_DIR}/python_packages/python")
set(CONFIG_PY_DIR "${CMAKE_INSTALL_PREFIX}/python_packages/python")
configure_file(py2mlir.py.in ${PROJECT_BINARY_DIR}/installFiles/py2mlir.py @ONLY)
install(PROGRAMS ${PROJECT_BINARY_DIR}/installFiles/py2mlir.py DESTINATION bin)