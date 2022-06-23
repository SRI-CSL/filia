#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

from mlir.ir import Type, Context

__all__ = [
    'PythonType',
    'ValueType',
    'CellType',
    'ScopeType'
]

class PythonType(Type):
  @staticmethod
  def isinstance(type: Type) -> bool: ...

class ValueType(PythonType):
  @staticmethod
  def isinstance(type: Type) -> bool: ...

  @staticmethod
  def get(context: Optional[Context] = None) -> ValueType: ...

class CellType(PythonType):
  @staticmethod
  def isinstance(type: Type) -> bool: ...

  @staticmethod
  def get(context: Optional[Context] = None) -> ValueType: ...

class ScopeType(PythonType):
  @staticmethod
  def isinstance(type: Type) -> bool: ...

  @staticmethod
  def get(context: Optional[Context] = None) -> ValueType: ...