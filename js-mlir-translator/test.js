

q = { q: 'Q'}

const y = Object.create(q)
console.log(`y.q: ${y.q}`)


y.p = 'P'

const x = Object.create(null)
console.log(`x.p: ${x.p}`)
console.log(`x.q: ${x.q}`)

