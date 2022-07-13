# This tests scoping rules of a with statement
import sys
import traceback as tb

class F:
  def __init__(self, e):
    self._expectExcept = e
    self._entered = False
    self._exited = False
  def __del__(self):
    assert(self._entered)
    assert(self._exited)

  def __enter__(self):
    self._entered = True
  def __exit__(self, exc_type, exc_value, traceback):
    assert(self._entered)
    self._exited = True
    if self._expectExcept:
      assert(exc_type == E)
      assert(exc_value != None)
      assert(type(exc_value) == E)
      # Catch exception
      return True
    else:
      assert(exc_type  == None)
      assert(exc_value == None)
      assert(traceback == None)

class E(Exception):
    pass

# Check f is undefined
try:
  f
  assert(False)
except NameError:
  pass

# Check g is entered/exited in normal case and g set to None
g = 1
with F(False) as g:
  pass
assert(g == None)

# Check assertions in F.__exit__
with F(True) as f:
  raise E()
assert(f == None) # This code is not actually unreachable.

