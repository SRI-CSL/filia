LLVM_INCLUDE=-I../../lorochi/llvm-install/include
LLVM_LIB=-L../../lorochi/llvm-install/lib

run : loadmlir
	./loadmlir ops.mlir

INCLUDE_FLAGS := $(LLVM_INCLUDE)
LD_FLAGS := -fuse-ld=lld $(LLVM_LIB) -lLLVMDemangle -lLLVMSupport \
  -lMLIRIR -lMLIRParser -lMLIRSupport \
  -lLLVMBitstreamReader -lLLVMBinaryFormat -lLLVMRemarks -lLLVMCore \
  -lMLIRStandard -lMLIRSideEffectInterfaces -lMLIRMemRef -lMLIRDialectUtils -lMLIRMemRefUtils \
  -lMLIRViewLikeInterface -lMLIRInferTypeOpInterface \
  -lMLIRTensor \
  -lMLIROpenMP -lMLIRLLVMIR -lMLIRControlFlowInterfaces -lMLIRDataLayoutInterfaces \
  -lcurses

loadmlir.o : loadmlir.cpp
	PATH="/usr/local/Cellar/llvm/13.0.0_2/bin:$(PATH)" \
	clang++ -c -o $@ -std=c++17 $(INCLUDE_FLAGS) $<

loadmlir : loadmlir.o
	PATH="/usr/local/Cellar/llvm/13.0.0_2/bin:$(PATH)" \
	clang++ -o $@ $< $(LD_FLAGS)

clean :
	rm -f loadmlir loadmlir.o

.PHONY : run clean
