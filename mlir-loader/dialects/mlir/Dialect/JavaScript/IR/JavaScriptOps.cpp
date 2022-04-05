#include "mlir/Dialect/JavaScript/JavaScriptOps.h"
//#include "TypeDetail.h"

#include "mlir/Dialect/JavaScript/JavaScriptTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/DialectImplementation.h"

//#include "llvm/ADT/StringRef.h"
//#include "llvm/ADT/Twine.h"
//#include "llvm/Support/MathExtras.h"
//#include <numeric>

#include<stdio.h>


using namespace mlir;
using namespace mlir::js;


#include "mlir/Dialect/JavaScript/JavaScriptDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/JavaScript/JavaScriptTypes.cpp.inc"
#define GET_OP_CLASSES
#include "mlir/Dialect/JavaScript/JavaScriptOps.cpp.inc"


/// RangeType prints as just "range".
static void print(ValueType rt, DialectAsmPrinter &os) { os << "value"; }

void JavaScriptDialect::printType(Type type, DialectAsmPrinter &os) const {
  print(type.cast<ValueType>(), os);
}

/// Parse a type registered to this dialect.
Type JavaScriptDialect::parseType(::mlir::DialectAsmParser &parser) const {
  // Parse the main keyword for the type.
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  MLIRContext *context = getContext();

  // Handle 'range' types.
  if (keyword == "value")
    return ValueType::get(context);

  parser.emitError(parser.getNameLoc(), "unknown Javascript type: " + keyword);
  return Type();

}


void JavaScriptDialect::initialize() {
  printf("Initializing JavaScript\n");
  addTypes<ValueType>();
  addOperations<
  #define GET_OP_LIST
  #include "mlir/Dialect/JavaScript/JavaScriptOps.cpp.inc"
      >();
}
