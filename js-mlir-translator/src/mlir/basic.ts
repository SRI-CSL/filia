import { WriteStream } from 'fs'
import {Op, Symbol, Region, BlockArgDecl, TypeAttr, ppCommas } from '../mlir'


export interface FunArg {
  name?: string
  type : TypeAttr
}

function ppArg(a: FunArg): string {
  if (a.name) {
    return `${a.name}: ${a.type}`
  } else {
    return `${a.type}`
  }
}

export function funType(args:Array<FunArg>, ret:TypeAttr): TypeAttr {
  return {
    toString() {
      return `(${ppCommas(args.map(ppArg))}) -> (${ret.toString()})`
    }
  }
}

function ppBlockArg(b:BlockArgDecl): string {
  return `${b.name}: ${b.type}`
}

export class Func extends Op {
  sym_name : Symbol
  type : TypeAttr
  body : Region|null

  constructor(sym_name: Symbol, type: TypeAttr, body: Region|null) {
    super()
    this.sym_name = sym_name
    this.type = type
    this.body = body
  }

  write(s: WriteStream, indent: string) {
    if (this.body && this.body.blocks.length > 0) {
      const blocks = this.body.blocks
      s.write(`${indent}func ${this.sym_name}${this.type} {\n`)
      if (blocks.length === 1) {
        const b = blocks[0]
        b.writeStmts(s, indent + '  ')
      } else {
        for (const b of blocks) {
          b.write(s, indent)
        }
      }
      s.write(`${indent}}\n`)
    } else {
      s.write(`${indent}func ${this.sym_name}${this.type}\n`)
    }
  }
}