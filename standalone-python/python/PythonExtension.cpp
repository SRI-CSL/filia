// The initial version of this file came from the LLVM project
// See https://llvm.org/LICENSE.txt for license information.

#include "Python-c/Dialects.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_pythonDialects, m) {
  //===--------------------------------------------------------------------===//
  // python dialect
  //===--------------------------------------------------------------------===//
  auto python_m = m.def_submodule("python");

  python_m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__python__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}
