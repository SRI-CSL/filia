# This tests how subscripts affects the local bindings.

x = [1]

# This validates that x is implicitly in outer scope
def test_nolocal():
    assert(x[0] == 1)
test_nolocal()

def test_subscript_local():
    x[0] = 2
    assert(x[0] == 2)

test_subscript_local()