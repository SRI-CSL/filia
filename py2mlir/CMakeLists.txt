# Build directory


#set(CONFIG_PY_DIR "${MLIR_BINARY_DIR}/python_packages/python")
#configure_file(py2mlir.py.in ${PROJECT_BINARY_DIR}/bin/py2mlir.py @ONLY)
#set(CONFIG_PY_DIR "${CMAKE_INSTALL_PREFIX}/python_packages/python")
#configure_file(py2mlir.py.in ${PROJECT_BINARY_DIR}/installFiles/py2mlir.py @ONLY)
#add_custom_command(TARGET test POST_BUILD
#                   COMMAND ${CMAKE_COMMAND} -E copy
#                       $<TARGET_FILE:test>
#                       installFiles/py2mlir2.py
#                   COMMENT "Copying 'test' library to '${COPY_TO_PATH}'")

add_custom_command(
        OUTPUT py2mlir.py
        COMMAND
        PY_DIR="${CMAKE_INSTALL_PREFIX}/python_packages/python"
        SOURCE=py2mlir.py.in
        TARGET="${CMAKE_CURRENT_BINARY_DIR}/py2mlir.py"
                cmake -P ${CMAKE_CURRENT_SOURCE_DIR}/configure_file.cmake
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS py2mlir.py.in)

add_custom_target(py2mlir ALL
        DEPENDS py2mlir.py)

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/py2mlir.py DESTINATION bin)