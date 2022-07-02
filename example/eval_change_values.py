# This is a simple test case to show that eval can mutate variables
# in scope through assignment operator.
x = 1
assert(x == 1)
assert(eval('(x := 3) + 1') == 4)
assert(x == 3)