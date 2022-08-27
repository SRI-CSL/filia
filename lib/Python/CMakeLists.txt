#set(CMAKE_CXX_STANDARD 17)

add_mlir_dialect_library(MLIRPython
        PythonDialect.cpp
        PythonOps.cpp
        PythonTypes.cpp
        FunctionValueTransitionMap.cpp
        PythonDomain.cpp
        PythonLoadStorePass.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Python

        DEPENDS
        MLIRPythonOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
target_compile_features(obj.MLIRPython PRIVATE cxx_std_17)
