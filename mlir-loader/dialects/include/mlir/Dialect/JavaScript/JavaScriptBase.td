//===- JavaScriptBase.td - Base definitions for math dialect ------*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef JAVASCRIPT_BASE
#define JAVASCRIPT_BASE
include "mlir/IR/OpBase.td"
def JavaScript_Dialect : Dialect {
  let name = "js";
  let summary = "A high-level dialect for analyzing JavaScript.";
  let description = [{
    The JavaScript dialect holds types and operations specific to JavaScript
    programs.
  }];
  let cppNamespace = "::mlir::js";

//  let useDefaultAttributePrinterParser = false;
//  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

#endif // JAVASCRIPT_BASE
