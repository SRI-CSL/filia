llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(PYTHON_TEST_DEPENDS
        FileCheck count not
        standalone-capi-test
        script-opt
        )
if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND PYTHON_TEST_DEPENDS PythonDialectModules)
endif()

add_lit_testsuite(check-python "Running the Python dialect regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${PYTHON_TEST_DEPENDS}
        )
set_target_properties(check-python PROPERTIES FOLDER "Tests")

add_lit_testsuites(PYTHON ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${PYTHON_TEST_DEPENDS})

add_subdirectory(CAPI)
