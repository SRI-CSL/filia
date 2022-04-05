// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef JAVASCRIPT_JAVASCRIPTOPS_H
#define JAVASCRIPT_JAVASCRIPTOPS_H

#include "JavaScript/JavaScriptTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "JavaScript/JavaScriptOps.h.inc"

#include "JavaScript/JavaScriptOpsEnums.h.inc"

#endif // JAVASCRIPT_JAVASCRIPTOPS_H