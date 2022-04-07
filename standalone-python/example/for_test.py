a = [1, 2, 3]
i = a.__iter__()
assert(i.__next__() == 1)
assert(i.__next__() == 2)
assert(i.__next__() == 3)
try:
        i.__next__()
        assert(False)
except StopIteration:
        pass