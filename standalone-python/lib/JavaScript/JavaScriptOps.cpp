// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "JavaScript/JavaScriptOps.h"
#include "JavaScript/JavaScriptDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "JavaScript/JavaScriptOps.cpp.inc"

#include "JavaScript/JavaScriptOpsEnums.cpp.inc"