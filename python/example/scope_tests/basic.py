# This tests semantics
def test_use_local_before_bound():
    # This fails because x is not bound.
    try:
        print(x)
        assert(False)
    except UnboundLocalError:
        pass
    x = 3
    assert(x)

# This tests use of the nonlocal function
def test_nonlocal():
    def inner():
        nonlocal x
        assert(x == 1)
        x=3

    # This makes x local
    x=1
    inner()
    assert(x == 3)

x=1
test_use_local_before_bound()
x=2



# Test
def test_global():
    global x
    assert(x == 2)
    x = 3

test_global()
assert(x == 3)