# This file reads from a file and passes the output to operator.
# It is intended as a information flow example because untrusted data
# flows into eval.

with open('example/op.txt') as f:
    operator = f.readline()
x = 3
result = eval(f"x {operator} 2")  # noqa: P204
print(result)
