import { WriteStream } from 'fs'
import { Op, TypeAttr, Value, ppCommas } from '../mlir'
import * as mlir from '../mlir'

export const ValueType : TypeAttr = {
  toString() {
    return "!js.value"
  }
}

export const LocalsType : TypeAttr = {
  toString() {
    return "!js.locals"
  }
}

// The Environment record specification type
// https://tc39.es/ecma262/#sec-environment-records
export const EnvironmentRecord : TypeAttr = {
  toString() {
    return "!js.environment_record"
  }
}

// Constructs a JavaScript "null" value
export class Null extends Op {
  constructor(private readonly ret: Value) {
    super()
  }

  toString() {
    return `${this.ret} = js.null : ${ValueType}\n`
  }
}

// Constructs a JavaScript "undefined" value
export class Undefined extends Op {
  constructor(private readonly ret: Value) {
    super()
  }

  toString() {
    return `${this.ret} = js.undefined : ${ValueType}`
  }
}

// Constructs a JavaScript "undefined" value
export class Number extends Op {
  constructor(private readonly ret: Value, private readonly val: number) {
    super()
  }

  toString() {
    return `${this.ret} = js.number ${this.val} : ${ValueType}`
  }
}

/*
export class MkArgListOp extends Op {
  constructor(ret:Value) {
    super()
  }
  write(s: WriteStream) {
    // FIXME
  }
}
*/

export interface CallArg {
  value: Value
  spread: boolean
}

function ppCallArg(a:CallArg): string {
  return `${a.value}` // FIXME
}

export class CallOp extends Op {
  constructor(readonly ret:Value, readonly callee:Value, readonly argList:CallArg[], readonly optional: boolean) {
    super()
  }
  toString(): Value {
    const args = ppCommas(this.argList.map((a) => a.value))
    const attrs : string[] = []
    if (this.optional) {
      attrs.push('js.optional')
    }
    const spreadIndices : string[] = []
    for (let i = 0; i < this.argList.length; ++i) {
      if (this.argList[i].spread) {
        spreadIndices.push(`${i}`)
      }
    }
    if (spreadIndices.length > 0) {
      attrs.push(`js.spread = [${ppCommas(spreadIndices)}]`)
    }
    const attrString = attrs.length > 0 ? ` {${ppCommas(attrs)}}` : ''
    return `${this.ret} = js.call ${this.callee}(${args}) ${attrString} : ${ValueType}`
  }
}

function escape(x:string): string {
  return `'${x}'` // FIXME
}

/** Construct a template literal */
export class TemplateLiteral extends Op {
  constructor(readonly ret:Value, readonly quasis: string[], readonly args: Value[]) {
    super()
  }

  toString(): Value {
    return `${this.ret} = js.template ${ppCommas(this.quasis.map(escape))} (${ppCommas(this.args)})`
  }
}

/*
export class ObjectExpression extends Op {
  constructor(r:Value) { // FIXME
    super()
  }

  write(s: WriteStream) {
    // FIXME
  }
}
*/

// This updates the local bindings to assign a name to a value.
export class EmptyLocals extends mlir.Op {
  constructor(readonly ret:Value) {
    super()
  }

  toString(): string {
    return `${this.ret} = js.empty_locals : ${LocalsType}`
  }
}

export type VarKind = "var" | "let" | "const";

// This updates the local bindings to assign a name to a value.
export class LocalDecl extends mlir.Op {
  constructor(readonly ret:Value, readonly map:Value, readonly kind: VarKind, readonly name: string, readonly val : Value|undefined) {
    super()
  }

  toString(): Value {
    return `${this.ret} = js.local_decl ${this.map}, ${this.kind}, ${this.name}, ${this.val} : ${LocalsType}`
  }
}

// This updates the local bindings to assign a name to a value.
export class LocalSet extends mlir.Op {
  constructor(readonly ret : Value, readonly map : Value, readonly name : string, readonly val : Value) {
    super()
  }

  toString(): string {
    return `${this.ret} = js.local_set ${this.map}, "${this.name}", ${this.val} : ${LocalsType}`
  }
}

// This returns the value associated with a name in bindings.
export class LocalGet extends mlir.Op {
  constructor(readonly ret:Value, readonly map:Value, readonly name: string) {
    super()
  }

  toString(): string {
    return `${this.ret} = js.local_get ${this.map}, "${this.name}" : ${LocalsType}`
  }
}

// This takes a Javascript value and returns a true bit if it is truthy.
// A value is truthy if it is not false, 0, -0, 0n, "", null, undefined, or NaN
export class Truthy extends mlir.Op {
  constructor(readonly ret:Value, readonly value:Value) {
    super()
  }

  toString(): string {
    return `${this.ret} = js.truthy ${this.value} : i1`
  }
}

export class StringLit extends mlir.Op {
  constructor(readonly ret:Value, readonly value:string) {
    super()
  }
  toString(): string {
    return `${this.ret} = js.string ${escape(this.value)} : i1`
  }
}

export class GetProperty extends mlir.Op {
  constructor(readonly ret:Value, readonly obj:Value, readonly prop:string) {
    super()
  }
  toString(): string {
    return `${this.ret} = js.get_property ${this.obj} ${escape(this.prop)} : i1`
  }
}
