# This tests how import affects the local bindings.

#  Define math variable for tests below
math = 1
def test_noimport_local():
    assert(math == 1)
test_noimport_local()

def test_import_local():
    # Math is imported below so it should be unbound here
    try:
        print(math)
        assert(False)
    except UnboundLocalError:
        pass

    # Import math and make sure it is local
    import math
    assert(type(math).__name__ == "module")

test_import_local()