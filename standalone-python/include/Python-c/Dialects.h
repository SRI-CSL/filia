// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef PYTHON_C_DIALECTS_H
#define PYTHON_C_DIALECTS_H

#include "mlir-c/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Python, python);

MLIR_CAPI_EXPORTED bool mlirTypeIsAPythonType(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAPythonValueType(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirPythonValueTypeGet(MlirContext ctx);

#ifdef __cplusplus
}
#endif

#endif // PYTHON_C_DIALECTS_H
