export interface Op {
  write(s:NodeJS.WriteStream)
}

/** Identifier for a block.  This should be a valid carot id */
export type BlockId = string

export interface BlockArg {

}

// A value is just an identifier in the name
export type Value = string

// A symbol reference (e.g. @foo)
export interface Symbol {
  type: "symbol"
  value: string
}

export function mkSymbol(value: string): Symbol {
  return { type: "symbol", value: value }
}



export interface Block {
  id: BlockId
  args : BlockArg[]
  statements: Op[]
}