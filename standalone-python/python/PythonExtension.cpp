// The initial version of this file came from the LLVM project
// See https://llvm.org/LICENSE.txt for license information.

#include "Python-c/Dialects.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

//===- DialectPDL.cpp - 'pdl' dialect submodule ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "Python/PythonTypes.h"

namespace py = pybind11;
using namespace llvm;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_pythonDialects, m) {
  m.doc() = "MLIR Python dialect.";
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

  auto pythonType = mlir_type_subclass(python_m, "PythonType", mlirTypeIsAPythonType);

  auto valueType = mlir_type_subclass(python_m, "ValueType", mlirTypeIsAPythonValueType);
  valueType.def_classmethod(
      "get",
      [](py::object cls, MlirContext ctx) {
        return cls(mlirPythonValueTypeGet(ctx));
      },
      "Get an instance of ValueType in given context.", py::arg("cls"),
      py::arg("context") = py::none());

  auto scopeType = mlir_type_subclass(python_m, "ScopeType", mlirTypeIsAPythonScopeType);
  scopeType.def_classmethod(
      "get",
      [](py::object cls, MlirContext ctx) {
        return cls(mlirPythonScopeTypeGet(ctx));
      },
      "Get an instance of ScopeType in given context.", py::arg("cls"),
      py::arg("context") = py::none());

}
