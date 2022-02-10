import * as mlir from '../mlir'

// Constructs a JavaScript "undefined" value
export class Undefined implements mlir.Op {
  constructor(ret:mlir.Value) {

  }

  write(s: NodeJS.WriteStream) {
    // FIXME
  }
}

// Constructs a JavaScript "null" value
export class Null implements mlir.Op {
  constructor(ret:mlir.Value) {

  }

  write(s: NodeJS.WriteStream) {
    // FIXME
  }
}

export class MkArgListOp implements mlir.Op {
  constructor(ret:mlir.Value) {

  }
  write(s: NodeJS.WriteStream) {
    // FIXME
  }
}

export class CallOp implements mlir.Op {
  constructor(r:mlir.Value, callee:mlir.Value, argList:mlir.Value) {
  }
  write(s: NodeJS.WriteStream) {
    // FIXME
  }
}

export class NewOp implements mlir.Op {
  constructor(r:mlir.Value, callee:mlir.SymbolRefId, argList:mlir.Value) {
  }
  write(s: NodeJS.WriteStream) {
    // FIXME
  }
}

export class TemplateLiteral implements mlir.Op {
  constructor(r:mlir.Value) { // FIXME
  }

  write(s: NodeJS.WriteStream) {
    // FIXME
  }
}

export class ObjectExpression implements mlir.Op {
  constructor(r:mlir.Value) { // FIXME
  }

  write(s: NodeJS.WriteStream) {
    // FIXME
  }
}
