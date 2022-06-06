def f(c, x, y):
    if c:
        return x
    else:
        return y

try:
    f(True, 1)
    assert(False)
except TypeError:
    pass

assert(f(True, 1, 2) == 1)
assert(f(y=3, x=1, c=False) == 3)
try:
    f(True, 1, y=3, c=True)
    assert(False)
except TypeError: # Multiple values for argument 'c'
    pass