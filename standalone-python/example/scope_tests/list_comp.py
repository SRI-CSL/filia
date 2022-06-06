
x = [1,2]
y = [[1,2]]

# Test product for loop
assert([i+j for i in x for j in x] == [2,3,3,4])

print([j for x in y for j in x])

# Test binding
assert([j for x in y for j in x] == [1,2])

# Test local binding
assert([x+1 for x in x] == [2,3])

print([x for a in x for x in y] == [[1,2], [1,2]])