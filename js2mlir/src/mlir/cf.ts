import { BlockId, ppCommas, TerminalOp, TypeAttr, Value } from '../mlir'

export class BlockArgValue {
  value: Value
  type: TypeAttr
}

export class BlockTarget {
  id: BlockId
  args: BlockArgValue[]
}

function ppBlockArgValue(a:BlockArgValue): string {
  return `${a.value} : ${a.type}`
}

function ppBlockTarget(tgt: BlockTarget): string {
  return `${tgt.id.label}(${ppCommas(tgt.args.map(ppBlockArgValue))})`
}

export class BranchOp extends TerminalOp {
  constructor(readonly target:BlockTarget) {
    super()
  }

  toString() {
    return `cf.br $${ppBlockTarget(this.target)}`
  }
}

export class CondBranchOp extends TerminalOp {
  constructor(private test: Value, private trueTarget: BlockTarget, private falseTarget: BlockTarget) {
    super()
  }

  toString() {
    return `cf.cond_br ${this.test}, ${ppBlockTarget(this.trueTarget)}, ${ppBlockTarget(this.falseTarget)}`
  }
}