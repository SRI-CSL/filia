import { WriteStream } from 'fs'
import { Op, TypeAttr, Value } from '../mlir'
import * as mlir from '../mlir'

export const ValueType : TypeAttr = {
  toString() {
    return "!js.value"
  }
}

// The Environment record specification type
// https://tc39.es/ecma262/#sec-environment-records
export const EnvironmentRecord : TypeAttr = {
  toString() {
    return "!js.environment_record"
  }
}


// Constructs a JavaScript "undefined" value
export class Undefined extends Op {
  private ret:Value

  constructor(ret: Value) {
    super()
    this.ret = ret
  }

  write(s: WriteStream, indent: string) {
    s.write(`${indent}  ${this.ret} = js.undefined : !js.value\n`)
  }
}

// Constructs a JavaScript "null" value
export class Null extends Op {
  constructor(ret:Value) {
    super()
  }

  write(s: WriteStream) {
    // FIXME
  }
}

export class MkArgListOp extends Op {
  constructor(ret:Value) {
    super()
  }
  write(s: WriteStream) {
    // FIXME
  }
}

export class CallOp extends Op {
  constructor(r:Value, callee:Value, argList:Value) {
    super()
  }
  write(s: WriteStream) {
    // FIXME
  }
}

export class NewOp extends Op {
  constructor(r:Value, callee:Value, argList:Value) {
    super()
  }
  write(s: WriteStream) {
    // FIXME
  }
}

export class TemplateLiteral extends Op {
  constructor(r:Value) { // FIXME
    super()
  }

  write(s: WriteStream) {
    // FIXME
  }
}

export class ObjectExpression extends Op {
  constructor(r:Value) { // FIXME
    super()
  }

  write(s: WriteStream) {
    // FIXME
  }
}

export class GetIdentifierReference extends Op {
  constructor(r:Value, e: Value, name: String, strict: boolean) {
    super()
  }

  write(s: WriteStream, indent: string) {
      //FIXME
      throw new Error('GetIdentifierReference.write not implemented')
  }
}

export class PutValue extends mlir.Op {
  constructor(lref:Value, rval : Value) {
    super()
  }

  write(s: WriteStream, indent: string) {
      //FIXME
      throw new Error('PutValue.write not implemented')
  }
}

