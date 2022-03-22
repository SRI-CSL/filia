//===- Dialects.cpp - CAPI for dialects -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Python-c/Dialects.h"

#include "Python/PythonDialect.h"
#include "Python/PythonTypes.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Python, python, mlir::python::PythonDialect)

bool mlirTypeIsAPythonType(MlirType type) {
  return unwrap(type).isa<python::PythonType>();
}

bool mlirTypeIsAPythonValueType(MlirType type) {
  return unwrap(type).isa<python::ValueType>();
}

MlirType mlirPythonValueTypeGet(MlirContext ctx) {
  return wrap(python::ValueType::get(unwrap(ctx)));
}