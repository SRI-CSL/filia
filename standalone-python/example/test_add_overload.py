class Add:
  def __add__(self, y):
    print('Hello from __add__')
    return self

class AddThrow:
  def __add__(self, y):
    raise Exception('Except')

class NoAdd:
  def __init__(self):
    pass

try:

  x = AddThrow()
  x + x
  assert(False)
except Exception:
  pass

a = Add()
n = NoAdd()
a + n
n + a

# n + x
