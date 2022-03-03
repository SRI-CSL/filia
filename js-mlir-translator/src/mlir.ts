import {WriteStream} from 'fs'

/** Identifier for a block.  This should be a valid carot id */
export interface BlockId {
  label: string
}

// A value is just an identifier in the name
export type Value = string

export abstract class Op {

  isTerminal(): boolean { return false }

  abstract write(s: WriteStream, indent: string)
}

export abstract class TerminalOp extends Op {

  isTerminal(): boolean { return true }

  abstract successors(): BlockId[]

  // Set arguments to successor block.
  //
  // Used to we can defer successor value assignment to
  // after a defuse pass.
  abstract setSuccessorArgs(index: number, values: Value[])
}


export interface BlockArg {
  name: Value
  type: string
}


// A symbol reference (e.g. @foo)
export interface Symbol {
  type: "symbol"
  // Symbol value (without leading `@`)
  name: string
  toString(): string
}

export function symbolValue(s:Symbol): string {
  return `@${s.name}`
}

export function mkSymbol(name: string): Symbol {
  return { type: "symbol", name: name, toString: () => `@${name}` }
}

function ppArg(b:BlockArg):string {
  return `${b.name}: ${b.type}`
}

export class Block {
  args : BlockArg[]

  constructor(readonly id: BlockId, readonly statements: Op[]) {
    this.args = []
  }

  setBlockArgs(args : BlockArg[]) {
    this.args = args
  }

  writeStmts(s: WriteStream, indent: string) {
    for (const op of this.statements) {
      op.write(s, indent)
    }
  }

  terminal(): TerminalOp {
    return this.statements[this.statements.length-1] as TerminalOp
  }


  successors():BlockId[] {
    const t = this.terminal()
    return t.successors()
  }

  write(s: WriteStream, indent: string) {
    var args:string
    if (this.args.length > 0) {
      args = `(${ppArg(this.args[0])}`
      for (var i = 1; i < this.args.length; ++i) {
        args = `${args}, ${ppArg(this.args[i])}`
      }
      args = `${args})`
    } else {
      args = ''
    }
    s.write(`${this.id}${args}:\n`)
    this.writeStmts(s, `${indent}  `)
  }
}

export interface Region {
  blocks: Block[]
}

// A MLIR type
export interface TypeAttr {
  toString():string
}
