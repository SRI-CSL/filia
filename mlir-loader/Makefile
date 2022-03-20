LLVM_INCLUDE=-I../../llvm-install/include
LLVM_LIB=-L../../llvm-install/lib
MLIR_TBLGEN=../../llvm-install/bin/mlir-tblgen

#LLVM_INCLUDE=-I$(HOME)/opt/llvm/include


INCLUDE_FLAGS := $(LLVM_INCLUDE)

LD_FLAGS := -fuse-ld=lld $(LLVM_LIB) -lLLVMDemangle -lLLVMSupport \
  -lMLIRIR -lMLIRParser -lMLIRSupport \
  -lLLVMBitstreamReader -lLLVMBinaryFormat -lLLVMRemarks -lLLVMCore \
  -lMLIRStandard -lMLIRSideEffectInterfaces -lMLIRMemRef -lMLIRDialectUtils -lMLIRMemRefUtils \
  -lMLIRViewLikeInterface -lMLIRInferTypeOpInterface \
  -lMLIRTensor \
  -lMLIROpenMP -lMLIRLLVMIR -lMLIRControlFlowInterfaces -lMLIRDataLayoutInterfaces \
  -lcurses

OBJS := dist/lib/loadmlir.o dist/lib/JavaScriptOps.o

CXX := /usr/local/Cellar/llvm/13.0.0_2/bin/clang++

INCDIR := dialects/include/mlir/Dialect/JavaScript
DISTDIR := dist/include/mlir/Dialect/JavaScript

loadmlir : $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LD_FLAGS)

dist/lib/loadmlir.o : loadmlir.cpp $(INCDIR)/JavaScriptOps.h $(INCDIR)/JavaScriptTypes.h $(DISTDIR)/JavaScriptDialect.h.inc  $(DISTDIR)/JavaScriptOps.h.inc  $(DISTDIR)/JavaScriptTypes.h.inc
	mkdir -p dist/lib
	$(CXX) -c -o $@ -std=c++17 -Idist/include $(INCLUDE_FLAGS) -Idialects/include -Idist/include $<

dist/lib/JavaScriptOps.o : dialects/mlir/Dialect/JavaScript/IR/JavaScriptOps.cpp $(INCDIR)/JavaScriptOps.h $(INCDIR)/JavaScriptTypes.h $(DISTDIR)/JavaScriptDialect.h.inc $(DISTDIR)/JavaScriptDialect.cpp.inc $(DISTDIR)/JavaScriptOps.h.inc $(DISTDIR)/JavaScriptOps.cpp.inc $(DISTDIR)/JavaScriptTypes.cpp.inc
	mkdir -p dist/lib
	$(CXX) -c -o $@ -std=c++17 -Idist/include $(INCLUDE_FLAGS) -Idialects/include -fno-rtti $<

tblgen : dialects/include/mlir/Dialect/JavaScript/JavaScriptBase.td
	mkdir -p dist/include/mlir/Dialect/JavaScript
	$(MLIR_TBLGEN) --gen-typedef-defs $(LLVM_INCLUDE) $<

.PHONY : tblgen

$(DISTDIR)/JavaScriptDialect.h.inc : dialects/include/mlir/Dialect/JavaScript/JavaScriptBase.td
	mkdir -p dist/include/mlir/Dialect/JavaScript
	$(MLIR_TBLGEN) --gen-dialect-decls $(LLVM_INCLUDE) -o $@ $<

$(DISTDIR)/JavaScriptDialect.cpp.inc : dialects/include/mlir/Dialect/JavaScript/JavaScriptBase.td
	mkdir -p dist/include/mlir/Dialect/JavaScript
	$(MLIR_TBLGEN) --gen-dialect-defs $(LLVM_INCLUDE) -o $@ $<


$(DISTDIR)/JavaScriptTypes.h.inc : dialects/include/mlir/Dialect/JavaScript/JavaScriptBase.td
	mkdir -p dist/include/mlir/Dialect/JavaScript
	$(MLIR_TBLGEN) --gen-typedef-decls $(LLVM_INCLUDE) -o $@ $<

$(DISTDIR)/JavaScriptTypes.cpp.inc : dialects/include/mlir/Dialect/JavaScript/JavaScriptBase.td
	mkdir -p dist/include/mlir/Dialect/JavaScript
	$(MLIR_TBLGEN) --gen-typedef-defs $(LLVM_INCLUDE) -o $@ $<

gen-op-decls :
	$(MLIR_TBLGEN) --gen-op-decls $(LLVM_INCLUDE)  dialects/include/mlir/Dialect/JavaScript/JavaScriptBase.td

$(DISTDIR)/JavaScriptOps.h.inc : $(INCDIR)/JavaScriptOps.td $(INCDIR)/JavaScriptBase.td
	mkdir -p dist/include/mlir/Dialect/JavaScript
	$(MLIR_TBLGEN) --gen-op-decls $(LLVM_INCLUDE) -Idialects/include -o $@ $<

$(DISTDIR)/JavaScriptOps.cpp.inc : $(INCDIR)/JavaScriptOps.td $(INCDIR)/JavaScriptBase.td
	mkdir -p dist/include/mlir/Dialect/JavaScript
	$(MLIR_TBLGEN) --gen-op-defs $(LLVM_INCLUDE) -Idialects/include -o $@ $<

run : loadmlir
	./loadmlir ops.mlir

clean :
	rm -rf loadmlir loadmlir.o dist

.PHONY : run clean
