const fs = require('fs')
operator = fs.readFileSync('example/op.txt', 'utf8')
x = 3
result = eval(`x ${operator} 2`)
console.log(result)