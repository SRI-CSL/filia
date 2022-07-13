# This tests how subscripts affects the local bindings.

x = [1, 1]

# This validates that x is implicitly in outer scope
def test_nolocal():
    assert(x[0] == 1)
test_nolocal()

# This tests that assigning to x as a subscript keeps it nonlocal
def test_subscript_local():
    x[1] = 2
    assert(x[1] == 2)

test_subscript_local()
assert(x[0] == 1)
assert(x[1] == 2)

# Test that assigning is local
def test_assign_local():
    x = [2, 1]
    assert(x[0] == 2)
assert(x[0] == 1)