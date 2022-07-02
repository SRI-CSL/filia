add_mlir_dialect_library(MLIRJavaScript
        JavaScriptDialect.cpp
        JavaScriptOps.cpp
        JavaScriptTypes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/JavaScript

        DEPENDS
        MLIRJavaScriptOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)