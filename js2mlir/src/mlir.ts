import {stat, WriteStream} from 'fs'

/** Identifier for a block.  This should be a valid carot id */
export interface BlockId {
  label: string
}

// A value is just an identifier in the name
export type Value = string

export abstract class Op {

  isTerminal(): boolean { return false }

  write(s: WriteStream, indent: string) {
    s.write(`${indent}${this.toString()}\n`)
  }
}

export abstract class TerminalOp extends Op {
  isTerminal(): boolean { return true }
}


export interface BlockArgDecl {
  name: Value
  type: TypeAttr
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

function ppArg(b:BlockArgDecl):string {
  return `${b.name}: ${b.type}`
}

export class Block {
  constructor(readonly id: BlockId, readonly args: BlockArgDecl[], readonly statements: Op[]) {

    if (statements.length == 0) {
      throw new Error(`Empty statements to add block`)
    }
    const n = statements.length-1
    for (var i = 0; i < n; ++i) {
      if (statements[i].isTerminal()) {
        throw new Error(`Internal statement ${i} of ${n} is terminal: ${statements[i]}`)
      }
    }

    const term = statements[n]

    if (!term.isTerminal()) {
      throw new Error(`Final statement ${term.constructor} is not terminal.`)
    }
  }

  /*
  setBlockArgs(args : BlockArg[]) {
    this.args = args
  }
  */

  writeStmts(s: WriteStream, indent: string) {
    for (const op of this.statements) {
      op.write(s, indent)
    }
  }


  write(s: WriteStream, indent: string) {
    var args:string
    if (this.args.length > 0) {
      args = `(${ppCommas(this.args.map(ppArg))})`
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

// Pretty print a comma separated list.
export function ppCommas(args:any[]):string {
  if (args.length == 0) {
    return ""
  }
  var r = args[0]
  for (var i = 1; i < args.length; ++i) {
    r = `${r}, ${args[i]}`
  }
  return r
}