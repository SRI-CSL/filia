

for i in range(4):
  print(i)

print(i)

k=4
x = [k for j in range(k) for k in range(j)]
try:
  print(j)
  assert(False)
except NameError:
  pass

print(x)