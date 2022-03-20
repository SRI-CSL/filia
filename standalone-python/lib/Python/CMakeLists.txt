add_mlir_dialect_library(MLIRPython
        PythonDialect.cpp
        PythonOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Python

        DEPENDS
        MLIRPythonOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
