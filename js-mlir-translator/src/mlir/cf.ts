import { BlockArg, BlockId, Op, TerminalOp, Value } from '../mlir'
import { WriteStream } from 'fs'


export class BranchOp extends TerminalOp {
  private targetArgs: Value[] = []

  constructor(readonly target:BlockId) {
    super()
  }

  successors(): BlockId[] {
    return [this.target]
}

  setSuccessorArgs(index: number, values: Value[]) {
    if (index === 0) {
      this.targetArgs = values
    } else {
      throw new Error(`Invalid index ${index}.`)
    }
  }

  write(s: WriteStream) {
    throw new Error(`Write BranchOp not implemented`)
  }
}

export class CondBranchOp extends TerminalOp {
  private trueArgs: Value[] = []
  private falseArgs: Value[] = []

  constructor(private test: Value, private trueTarget: BlockId, private falseTarget: BlockId) {
    super()
  }

  successors(): BlockId[] {
      return [this.trueTarget, this.falseTarget]
  }

  setSuccessorArgs(index: number, values: Value[]) {
    if (index === 0) {
      this.trueArgs = values
    } else if (index === 1) {
      this.falseArgs = values
    } else {
      throw new Error(`Invalid index ${index}.`)
    }
  }

  write(s: WriteStream) {
    // FIXME
    throw new Error(`Write CondBranchOp not implemented`)
  }
}