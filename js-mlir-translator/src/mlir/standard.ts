import {TerminalOp, Value, TypeAttr } from '../mlir'

interface ReturnValue {
  value: Value
  type: TypeAttr
}

export class ReturnOp extends TerminalOp {
  constructor(readonly values: ReturnValue[]) {
    super()
  }

  toString() {
    let res : string = ''
    if (this.values.length > 0) {
      var valStr  = this.values[0].value
      var typeStr = this.values[0].type
      for (var i = 1; i < this.values.length; ++i) {
        valStr = `${valStr}, ${this.values[i].value}`
        typeStr = `${typeStr}, ${this.values[i].type}`
      }
      res = `${valStr} : ${typeStr}`
    }

    return `return ${res}`

  }
}
