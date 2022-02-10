export type Ident = string

// This returns true if the Javascript identifier may be used directly as an
// MLIR suffix-id
//
// N.B. A JavaScript identifier must start with a letter, underscore (_), or
// dollar sign ($). Subsequent characters can also be digits (0–9).  Javascript
// identifiers may include Unicode letters.
//
// MLIR value identifiers start with a '%' and are followed by a suffix-id constructed
// with the following grammar:
//   suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))
//   letter    ::= [a-zA-Z]
//   id-punct  ::= [$._-]
export function allowedIdent(x: string): boolean {
  return /^([A..Z]|[a..z]|[_$])([A..Z]|[a..z]|[_$]|[0..9])*$/.test(x)
}
