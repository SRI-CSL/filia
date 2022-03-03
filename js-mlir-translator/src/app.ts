import { assert } from 'console'
import * as esprima from 'esprima'
import * as estree from 'estree'
import { WriteStream } from 'fs'
import * as fs from 'fs'
import * as js from './javascript'
import * as mlir from "./mlir"
import * as mlirb from './mlir/basic'
import * as mlirjs from './mlir/javascript'
import * as cf from './mlir/cf'
import { ReturnOp  } from './mlir/standard'
import { BlockId, Block, Op, Value } from "./mlir"

import Heap from 'heap-js';

namespace Mlir {

  interface IsType {
    render(): string
  }

  class FunctionType implements IsType {
    inputs : Type[]
    results : Type[]
    render() {
      return "1";
    }
  }

  type Type = FunctionType

}

function assertNever(x: never):any {
  console.log(`Not never ${JSON.stringify(x)}`)
  throw new Error("Was not never: " + x);
}


interface Warning {
  loc:estree.SourceLocation|null|undefined,
  message: string
}

function formatWarning(w:Warning): string {
  if (!w.loc) {
    return `Unknown: ${w.message}`
  }
  const start = w.loc.start
  const end = w.loc.end
  if (start.line === end.line) {
    return `Ln ${start.line},Col ${start.column+1}-${end.column+1}: ${w.message}`
  } else {
    return `Ln ${start.line},Col ${start.column+1}-Ln ${end.line},Col ${end.column+1}: ${w.message}`
  }
}

class TranslationError extends Error {
  constructor(loc:estree.SourceLocation|null|undefined, message:string) {
    console.log(`Exception: ${formatWarning({loc: loc, message: message})}`)
    super(formatWarning({loc: loc, message: message}))
  }
}

interface Region {

}


type CaseTest=string|boolean|number

class Case {
  constructor(test: CaseTest, action:BlockId) {
    this.test = test
    this.action = action
  }
  readonly test: CaseTest
  readonly action: BlockId
}

class SwitchStmt extends mlir.TerminalOp {
  #successors: mlir.BlockId[]
  #caseArgs : Value[][]
  #defaultArgs : Value[]

  constructor(private cases:Case[], private defaultCase:BlockId) {
    super()
    this.#successors = []
    for (const c of cases) {
      this.#successors.push(c.action)
    }
    this.#successors.push(defaultCase)
    this.#caseArgs = new Array(this.cases.length)
  }

  successors(): mlir.BlockId[] {
    return this.#successors
  }

  setSuccessorArgs(index: number, values: Value[]) {
    if (index < this.cases.length) {
      this.#caseArgs[index] = values
    }
    this.#defaultArgs = values
  }

  write(s: WriteStream) {
    throw new Error('SwitchStmt unimplemented')
  }
}

class StdConstant extends mlir.Op {
  constructor(ret:mlir.Value, sym:mlir.Symbol) {
    // FIXME
    super()
  }
  write(s: WriteStream) {
    // FIXME
  }
}

class ThrowOp extends mlir.TerminalOp {
  constructor(v:mlir.Value) {
    super()
  }

  successors(): mlir.BlockId[] {
    return []
  }

  setSuccessorArgs(index: number, values: string[]) {
    throw new Error('ThrowOp has no successors.')
  }

  write(s: WriteStream) {
    // FIXME
  }
}

interface ClassInfo {
  readonly classConstructor:mlir.Symbol
}

// A global variable.
interface GlobalVar {
  type: "globalvar"
  varId: string
}


class Module {
  private symbolGen = new IdentGenerator('')
  private anonClassNum:number=0

  // Operations added so far.
  ops : Array<mlir.Op> = new Array()


  freshFunctionId(x: string|null):mlir.Symbol {
    let val = this.symbolGen.freshIdent(x ? x : `_F`)
    return mlir.mkSymbol(val)
  }

  addFunction(fun:FunctionContext) {
    this.ops.push(fun.getFunction())
  }

  uniqueClassName(name:estree.Identifier|null): string {
    if (name) {
      return name.name
    }
    return `class${this.anonClassNum++}`
  }

}

class IdentGenerator {
  prefix: string

  // Maps identifiers to number of times it has been used so we
  // can generate fresh numbers.
  private identIndexMap: Map<js.Ident, number> = new Map()

  // Generator for values not formed from identifiers.
  private nextValueIndex: number = 0

  constructor(prefix: string) {
    this.prefix = prefix
  }

  freshAnon(): Value {
    const val = `${this.prefix}${this.nextValueIndex}`
    this.nextValueIndex++;
    return val
  }

  // Return an unused value name derived from the name of the identifier.
  freshIdent(v: js.Ident): Value {
    console.log(`Fresh ident ${v} ${js.allowedIdent(v)}`)
    if (js.allowedIdent(v)) {
      const n = this.identIndexMap.get(v)
      if (n === undefined) {
        this.identIndexMap.set(v, 0)
        return `${this.prefix}${v}`
      } else {
        this.identIndexMap.set(v, n+1)
        return `${this.prefix}${v}.${n}`
      }
    } else {
      return this.freshAnon()
    }
  }

}

// A synthetic variable local to a function
interface FunctionVar {
  type: "funvar"
  varId: string
}

// Creates a map from blocks to their index.
function mkBlockIdMap(blocks:BlockInfo[]):Map<BlockId, number> {
  // Map from blocks ids to index in blocks[]
  const blockMap: Map<BlockId, number> = new Map()
  for (var i=0; i < blocks.length; ++i) {
    blockMap.set(blocks[i].id, i)
  }
  return blockMap
}

// Performs a reverse post-order traversal of blocks and returns new blocks
// This tries to preserve the order of blocks while ensuring a post-order constraint
// is satisfied.
function reversePostOrder(blocks:BlockInfo[],
                          blockMap:Map<BlockId, number>):BlockInfo[] {
  assert(blocks.length > 0)

  // Array for result.
  const r:Array<BlockInfo> = []

  // Create frontier of nods to visit
  const frontier = new Heap()
  frontier.push(0)
  // Capture if addded to frontier
  const added:boolean[] = new Array<boolean>(blocks.length)
  added[0] = true
  for (var i=1; i < blocks.length; ++i) {
    added[i] = false
  }

  var next:number|undefined
  // Keep running while we still have nodes on frontier
  while (next = frontier.pop() as any) {
    const b = blocks[next]
    r.push(b)
    b.stmts
    const term = b.stmts[b.stmts.length-1] as mlir.TerminalOp
    const succArray = term.successors()
    // Add blocks in reverse order
    for (var i = succArray.length-1; i >= 0; --i) {
      const idx = blockMap.get(succArray[i])
      if (!idx) throw new Error('Invalid block id')
      assert(idx)
      if (!added[idx]) {
        added[idx] = true
        frontier.push(idx)
      }
    }
  }

  return r
}

// Map from blocks to the variables they need.
type BlockInputMap = Map<BlockId, FunctionVar[]>

// Map from variables to their value when the block ends.
type VarValueMap = Map<FunctionVar, Value>

// Map from blocks to the variable values when block ends.
type BlockTermValueMap = Map<BlockId, VarValueMap>

function mkBlock(b:BlockInfo):Block {
  return new Block(b.id, b.stmts)
}

function populateSuccessorArgs(blockTermValues: BlockTermValueMap, blockInputVars: BlockInputMap, b:Block) {
  const valMap = blockTermValues.get(b.id)
  if (!valMap)
    throw new Error(`Could not find value map for block ${b.id}.`)
  const t = b.terminal()
  const suList = t.successors()
  for (var idx = 0; idx < suList.length; ++idx) {
    // Get identifier of successor
    const successorId = suList[idx]
    const inputs = blockInputVars.get(successorId)
    if (!inputs) throw new Error('Could not find input variables.')

    const args : Array<Value> = []
    for (const input of inputs) {
      const val = valMap.get(input)
      if (!val)
        throw new Error(`Could not find value for ${input}`)
      args.push(val)
    }
    t.setSuccessorArgs(idx, args)
  }
}

function intersect(doms: number[], b1: number, b2: number):number {
  while (true) {
    if (b1 > b2) {
      b1 = doms[b1]
    } else if (b2 > b1) {
      b2 = doms[b2]
    } else {
      return b1
    }
  }
}

type IDomArray = number[]

// predArray[i] contains predecessors of block[i]
type PredArray = number[][]


// Create a dominator
function computeIdoms(
     // predArray[i] contains predecessors of block[i]
    predArray: number[][]): IDomArray {

  assert (predArray.length > 0)
  assert (predArray.length)

  const doms : Array<number> = new Array(predArray.length)
  // First block is zero
  doms[0] = 0
  var changed:boolean
  do {
    changed = false
    for (var i = 1; i < predArray.length; ++i) {
      const preds = predArray[i]
      assert(preds.length > 0)
      var d0 = doms[i]
      var d = doms[i] ?? preds[0]
      for (var j = 1; j < preds.length; ++j) {
        d = intersect(doms, d, doms[j])
      }
      if (d0 != d) {
        d0 = d
        changed = true
      }
    }
  } while (changed)
  return doms
}

// Return true if all paths to tgt must go through src.
function isDominator(doms: IDomArray, src:number, tgt:number):boolean {
  while (src !== tgt) {
    const next = doms[tgt]
    if (next == tgt)
      return false
    tgt = next
  }
  return true
}

interface BlockInfo {
  id: BlockId,
  readVars: ReadVars,
  stmts:Op[],
  term:mlir.TerminalOp,
  assignedVars : AssignedVars
}

function blockReadClosure(blocks:BlockInfo[], succMap: number[][]): Array<Set<FunctionVar>> {

    // Compute what variables each block needs.
    var changed:boolean
    // Maps from block ids to the function variables that they need.
    const blockInputVarSet : Array<Set<FunctionVar>> = new Array(blocks.length)

    do {
      changed = false
      for (var i = blocks.length-1; i >= 0; --i) {
        const b = blocks[i]
        var bchanged = false
        var s = blockInputVarSet[i]
        if (!s) {
          s = new Set(b.readVars)
          bchanged = true
        }
        for (var j of succMap[i]) {
          var jread = blockInputVarSet[j]
          if (jread) {
            for (const v of jread) {
              if (!b.assignedVars.has(v) && !s.has(v)) {
                s.add(v)
                bchanged = true
              }
            }
          }
        }
        // Assign variables
        if (bchanged)
          blockInputVarSet[i] = s
      }
    } while (changed)
    return blockInputVarSet
}

// Information about a MLIR function being constructed by the translator.
class FunctionContext {
  // Symbol identifying function
  symbol: mlir.Symbol

  // Blocks to add  to the current function
  private readonly blocks:Array<BlockInfo> = new Array()

  // Identifer to use for next block.
  private nextBlockId:number = 0

  private gen = new IdentGenerator('%')

  constructor(symbol: mlir.Symbol) {
    this.symbol = symbol
  }

  freshBlockId():BlockId {
    const res=`^${this.nextBlockId}`
    this.nextBlockId += 1;
    return { label: res }
  }

  // Create a new synthetic variable scope to function
  freshVar(name?:js.Ident):FunctionVar {
    const gen = this.gen
    return { type: "funvar", varId: name ? gen.freshIdent(name) : gen.freshAnon() }
  }

  // Add block to end of blocks
  addBlock(info: BlockInfo) {
    const stmts = info.stmts
    if (stmts.length == 0) {
      throw new Error(`Empty statements to add block`)
    }
    const n = stmts.length-1
    for (var i = 0; i < stmts.length-1; ++i) {
      if (stmts[i].isTerminal()) {
        throw new Error('Internal statement is terminal.')
      }
    }

    const term = stmts[stmts.length-1]

    if (!term.isTerminal()) {
      throw new Error(`Final statement ${term.constructor} is not terminal.`)
    }

    this.blocks.push(info)
  }

  getFunction() : mlirb.Func {
    // Perform reverse post-order traversal to get new blocks.
    // This also prunes unreachable blocks.
    const postBlocks = reversePostOrder(this.blocks, mkBlockIdMap(this.blocks))
    assert(postBlocks.length > 0)

    const predArray: PredArray = [] // FIXME

    const doms = computeIdoms(predArray)

    // Perform argument analysis.
    // Compute what identifiers each block needs
    // Compute dominators
    // Compute map from blocks to their predecessors

    const argTypes : mlirb.FunArg[] = [] // FIXME
    //const argTypes = [{name: "args", type: mlirjs.ValueListType}]

    // Create map from block identifiers to index in postBlocks array
    const blockIdMap : Map<BlockId, number> = new Map()
    for (var i = 0; i < postBlocks.length; ++i) {
      blockIdMap.set(postBlocks[i].id, i)
    }

    // Create successor arrays based on indices in postBlocks array
    const succMap : number[][] = new Array(postBlocks.length)
    for (var i = 0; i < postBlocks.length; ++i) {
      const b = postBlocks[i]
      const succIds = b.term.successors()
      const succ = new Array(succIds.length)
      for (var j = 0; j < succIds.length; ++j) {
        var idx = blockIdMap.get(succIds[j])
        if (idx === undefined)
          throw new Error(`Unknown successor block ${succIds[j]}`)
        succ[j] = idx
      }
      succMap[i] = succ
    }

    // Maps from block ids to the function variables that they need.
    const blockInputVarSet : Array<Set<FunctionVar>> = blockReadClosure(postBlocks, succMap)

    const blockInputVars : BlockInputMap = new Map()
    for (var i = 0; i < postBlocks.length; ++i) {
      const b = postBlocks[i]
      const vars = blockInputVarSet[i]
      blockInputVars.set(b.id, [...vars])
    }

    // Map from blocks to the values of terminating blocks
    const blockTermValues : BlockTermValueMap = new Map() // FIXME

    const blocks = postBlocks.map(mkBlock)

    // Populate sucessor arguments
    for (const b of blocks) {
      populateSuccessorArgs(blockTermValues, blockInputVars, b)
    }

    const region = { blocks: blocks }

    const retType = mlirjs.ValueType
    return new mlirb.Func(this.symbol, mlirb.funType(argTypes, retType), region)
  }

}

type ReadVars =  Set<FunctionVar>
type AssignedVars = Map<FunctionVar, Value>

/// Information about the current block being generated.
class BlockContext {
  // Identifier for this block
  readonly id:BlockId

  // Statements for the current block.
  readonly statements:Array<Op> = new Array()

  // Variables read from previous block needed to compute block.
  readonly readVars : ReadVars = new Set()

  // Map from variables to their value in the block.
  readonly assignedVars : AssignedVars = new Map()

  // Number of values
  private valueCount : number = 0

  constructor(blockId: BlockId) {
    this.id = blockId
  }

  appendStatement(op: Op) {
    this.statements.push(op)
  }

  // Create a fresh value for the block internals.
  freshValue(): Value {
    return `%$${this.valueCount++}`
  }

  assign(lhs: Var, rhs:Value) {
    switch (lhs.type) {
      case "funvar":
        this.assignedVars.set(lhs, rhs)
        break
      case "globalvar":
        // FIXME
        throw new TranslationError(null, "Cannot write global variables yet.")
      default:
        assertNever(lhs)
    }
  }

  // Return an unused value name derived from the name of the identifier.
  /*
  freshIdentValue(v: Var): Value {
    if (typeof v === 'number') {
      return `%$${v}` // Make sure synthetic identifiers are distinct from freshIdent
    } else {
      return this.gen.freshIdent(v)
    }
  }
  */

  // Return a value associated with the given identifier.
  readVar(v: Var): Value {
    switch (v.type) {
      case "funvar":
        {
          var val = this.assignedVars.get(v)
          if (val === undefined) {
            val = v.varId
            this.readVars.add(v)
            this.assignedVars.set(v, val)
          }
          return val
        }
      case "globalvar":
        // FIXME
        throw new TranslationError(null, "Cannot read global variables yet.")
      default:
        return assertNever(v)
    }
  }

  /** Return a value equal to undefined. */
  undefinedJavascriptValue():Value {
    let v = this.freshValue()
    this.appendStatement(new mlirjs.Undefined(v))
    return v
  }
}


// Information about a block to jump to within this function
// to catch an exception
class CatchInfo {
  block: BlockId
  // Variable to assign exception to when jumping to block.
  var:FunctionVar
}

interface ScopeConfig {
  // New value to use for this (or undefined to copy from parent).
  inheritThis?:boolean
  // Method to use for invoking super
  superMethod?:Value|null
  // Start a fresh function (rather than use parent)
  newFunction?:FunctionContext
  // Append statements to parent block (default to create new block)
  newBlock?:BlockId
  continueBlock?:BlockId|null
  breakBlock?:BlockId|null
  catchBlock?:CatchInfo|null
}


type Var=FunctionVar|GlobalVar

class Scope {
  // Global warnings array
  readonly warnings: Array<Warning>;

  // Parent for lookup scope
  private readonly parent: Scope|null

  // Length of parent chain (used for building references)
  private readonly level: number

  module: Module

  // Information about function
  private function: FunctionContext

  // Information from current block.
  private block: BlockContext

  // Set of identifiers defined in this scope
//  private readonly identSet: Set<string>

  // Value to use when translating "this"
  thisVar: FunctionVar|null

  // Value that denotes function to invoke with super(...)
  private readonly superMethod:Value|null

  // Block to jump to for continue or null if a continue should not be allowed
  private continueBlock:BlockId|null

  // Block to jump to for break or null if a break should not be allowed
  private breakBlock:BlockId|null

  //
  private catchBlock:CatchInfo|null


  // Maps string to variables local to function or global symbols
  //private readonly identMap: Map<string, Var> = new Map()


  private readonly lexicalEnvironment: FunctionVar

  constructor(parent:Scope|null, config:ScopeConfig) {
    this.warnings = parent ? parent.warnings : new Array<Warning>()
    this.parent = parent
    this.level = parent ? parent.level + 1 : 0
    this.module = parent ? parent.module : new Module()
    if (config.newFunction) {
      this.function = config.newFunction
      this.lexicalEnvironment = this.function.freshVar('env')
    } else if (parent) {
      this.function = parent.function
      this.lexicalEnvironment = parent.lexicalEnvironment
    } else {
      this.function = new FunctionContext(this.module.freshFunctionId('main'))
      this.lexicalEnvironment = this.function.freshVar('env')
    }

    if (config.newBlock) {
      this.block = new BlockContext(this.function.freshBlockId())
    } else {
      if (!parent) {
        throw new Error('Expected parent when newBlock not set')
      }
      this.block = parent.block
    }

    this.thisVar
      = !parent ? this.function.freshVar('this')
      : config.inheritThis !== false ? parent.thisVar : null
    this.superMethod
      = config.superMethod !== undefined
      ? config.superMethod
      : parent ? parent.superMethod : null
    this.continueBlock
      = config.continueBlock !== undefined
      ? config.continueBlock
      : parent ? parent.continueBlock : null
    this.breakBlock
      = config.breakBlock !== undefined
      ? config.breakBlock
      : parent ? parent.breakBlock : null
    this.catchBlock
      = config.catchBlock !== undefined
      ? config.catchBlock
      : parent ? parent.catchBlock : null
  }

  addStatement(op:Op):void {
    this.block.appendStatement(op)
  }

  addWarning(loc:estree.SourceLocation|null|undefined, msg: string) {
    this.warnings.push({loc: loc, message: msg})
  }

  unhandledExpression(e: estree.Expression):Value {
    this.addWarning(e.loc, `Unhandled expression: ${e.type}`)
    return this.block.undefinedJavascriptValue()
  }


  unhandledPattern(p:estree.Pattern) {
    this.addWarning(p.loc,`Unhandled pattern: ${p.type}`)
  }

  /*
  addPatternIdent(p:estree.Pattern):Value {
    switch (p.type) {
      case 'Identifier':
        {
          const v = this.function.freshVar(p.name)
          this.block.appendStatement()
          this.lexicalEnvironment

          this.identMap.set(p.name, v)
          return this.block.readVar(v)
        }
      case 'ObjectPattern':
      case 'ArrayPattern':
      case 'RestElement':
      case 'AssignmentPattern':
      case 'MemberExpression':
        {
          const val = this.block.freshValue()
          this.unhandledPattern(p)
          return val
        }
      default:
        return assertNever(p)
    }
  }
  */

  translateFunctionExpression(e: estree.FunctionExpression) {
    const ident : estree.Identifier|null|undefined = e.id

    if (e.generator) {
      throw new TranslationError(e.loc, 'Generator functions are not yet supported.')
    }
    if (e.async) {
      throw new TranslationError(e.loc, 'Async functions are not yet supported.')
    }

    //for (const p of e.params) {
    //  const val = this.addPatternIdent(p) // Value
    //}

    this.translateBlockStatement(e.body)

    this.module.addFunction(this.function)
  }

  translateUnaryExpression(e: estree.UnaryExpression): Value {
    const a = this.translateExpression(e.argument)
    switch (e.operator) {
      case "-":
      case "+":
      case "!":
      case "~":
      case "typeof":
      case "void":
      case "delete":
        this.addWarning(e.loc, `Unsupported unary operator ${e.operator}.`)
        return this.block.undefinedJavascriptValue()
      default:
        return assertNever(e.operator)
    }
  }

  translateBinaryExpression(e: estree.BinaryExpression): Value {
    const l = this.translateExpression(e.left)
    const r = this.translateExpression(e.right)
    switch (e.operator) {
      case "==":
      case "!=":
      case "===":
      case "!==":
      case "<":
      case "<=":
      case ">":
      case ">=":
      case "<<":
      case ">>":
      case ">>>":
      case "+":
      case "-":
      case "*":
      case "/":
      case "%":
      case "**":
      case "|":
      case "^":
      case "&":
      case "in":
      case "instanceof":
        this.addWarning(e.loc, `Unsupported binary operator ${e.operator}.`)
        return this.block.undefinedJavascriptValue()
      default:
        return assertNever(e.operator)
    }
  }

  /*
  lookupVar(v: estree.Identifier): Var {
    var cur:Scope|null=this
    while (cur) {
      const val = cur.identMap.get(v.name)
      if (val === undefined) {
        cur = cur.parent
      } else {
        return val
      }
    }
    throw new TranslationError(v.loc, `Could not find identifier ${v.name}`)
  }
  */

  strictMode: boolean

  // 13.15.2 Runtime Semantics: Evaluation
  // https://tc39.es/ecma262/#sec-assignment-operators
  translateAssignmentExpression(e: estree.AssignmentExpression): Value {
    var v = this.translateExpression(e.right)
    switch (e.operator) {
      case "=":
        break
      case "+=":
      case "-=":
      case "*=":
      case "/=":
      case "%=":
      case "**=":
      case "<<=":
      case ">>=":
      case ">>>=":
      case "|=":
      case "^=":
      case "&=":
        this.addWarning(e.loc, `Unsupported assignment operator ${e.operator}.`)
        break
      default:
        return assertNever(e.operator)
    }
    const left = e.left
    switch (left.type) {
      case 'Identifier':
        const env = this.block.readVar(this.lexicalEnvironment)
        const lref = this.block.freshValue()
        this.block.appendStatement(new mlirjs.GetIdentifierReference(lref, env, left.name, this.strictMode))
        this.block.appendStatement(new mlirjs.PutValue(lref, v))
        break
      default:
        this.addWarning(e.loc, `Unsupported assignment left-hand-side ${left.type}.`)
        break
    }
    return v
  }

  translateLogicalExpression(e: estree.LogicalExpression):Value {
    this.translateExpression(e.left)
    this.translateExpression(e.right)
    switch (e.operator) {
      case "||":
      case "&&":
      case "??":
        this.addWarning(e.loc, `Unsupported logical expression ${e.operator}`)
        return this.block.undefinedJavascriptValue()
      default:
        return assertNever(e.operator)
    }
  }

  translateMemberExpression(e: estree.MemberExpression):Value {
    if (e.computed)
      this.addWarning(e.loc, 'Computed member expression not supported')
    if (e.optional)
      this.addWarning(e.loc, 'Optional member expression not supported')

    if (e.object.type === 'Super') {
      // FIXME
      throw new TranslationError(e.loc, 'Super value is undefined')
    }

    let o = this.translateExpression(e.object)

    switch (e.property.type) {
      case 'Identifier':
        const name = e.property.name
        return this.block.undefinedJavascriptValue()
      default:
        this.addWarning(e.loc, `property ${e.property.type} unsupported`)
        return this.block.undefinedJavascriptValue()
    }
  }

  private translateConditionalExpression(e: estree.ConditionalExpression): Value {
    const c = this.translateExpression(e.test)

    const v = this.function.freshVar()

    const trueId = this.function.freshBlockId()
    const falseId = this.function.freshBlockId()
    const nextBlock = this.function.freshBlockId()

    const trueScope = new Scope(this, {newBlock: trueId})
    const trueValue = trueScope.translateExpression(e.consequent)
    trueScope.block.assign(v, trueValue)
    trueScope.endBlock(new cf.BranchOp(nextBlock))

    const falseScope = new Scope(this, {newBlock: falseId})
    const falseValue = falseScope.translateExpression(e.alternate)
    falseScope.block.assign(v, falseValue)
    falseScope.endBlock(new cf.BranchOp(nextBlock))

    this.startFreshBlock(new cf.CondBranchOp(c, trueId, falseId), nextBlock)

    return this.block.readVar(v)
  }

  translateCallArguments(args: Array<estree.Expression | estree.SpreadElement>): Value {
    const argList = this.block.freshValue()
    for (const arg of args) {
      // FIXME add arguments
      if (arg.type === 'SpreadElement') {
        throw new TranslationError(arg.loc, 'Spread parameters are not supported')
      } else {
        this.translateExpression(arg)
      }
    }
    this.block.appendStatement(new mlirjs.MkArgListOp(argList))

    return argList
  }

  translateCallExpression(e: estree.CallExpression):Value {
    var callee:Value
    if (e.callee.type === 'Super') {
      if (!this.superMethod)
        throw new TranslationError(e.callee.loc, 'Super method is undefined')
      callee = this.superMethod
    } else {
      callee = this.translateExpression(e.callee)
    }
    const argList = this.translateCallArguments(e.arguments)

    const r = this.block.freshValue()
    this.block.appendStatement(new mlirjs.CallOp(r, callee, argList))
    return r
  }

  translateNewExpression(e: estree.NewExpression):Value {
    var callee:Value
    if (e.callee.type === 'Super') {
      throw new TranslationError(e.callee.loc, 'new super is not defined')
    } else {
      callee = this.translateExpression(e.callee)
    }
    const argList = this.translateCallArguments(e.arguments)

    const r = this.block.freshValue()
    this.block.appendStatement(new mlirjs.NewOp(r, callee, argList))
    return r
  }

  /**
   * Translate a template literal (e.g., `Hello ${var}`)
   */
  translateTemplateLiteral(l: estree.TemplateLiteral): Value {
    for (const exp of l.expressions) {
      this.translateExpression(exp)
    }
    const r = this.block.freshValue()
    this.block.appendStatement(new mlirjs.TemplateLiteral(r))
    return r // FIXME
  }


  translateObjectExpression(e:estree.ObjectExpression): Value {
    for (const p of e.properties) {
      switch (p.type) {
        case 'Property':
          {
            var val :Value
            switch (p.value.type) {
              case 'ObjectPattern':
              case 'ArrayPattern':
              case 'RestElement':
              case 'AssignmentPattern':
                throw new TranslationError(p.value.loc, `{p.value.type} pattern not allowed.`)
              default:
                val = this.translateExpression(p.value)
            }
            switch (p.key.type) {
              case 'Identifier':
                break
              default:
                throw new TranslationError(p.key.loc, `${p.key.type} object key not allowed.`)
            }
            break
          }
        case 'SpreadElement':
          throw new TranslationError(p.loc, 'SpreadElement not supported')
        default:
          assertNever(p)
      }
    }
    const r = this.block.freshValue()
    this.block.appendStatement(new mlirjs.ObjectExpression(r))
    return r // FIXME
  }

  translateLiteral(e: estree.Literal) {
    return this.unhandledExpression(e)
  }


  translateExpression(e: estree.Expression):Value {
    switch (e.type) {
      case 'ThisExpression':
        if (!this.thisVar)
          throw new TranslationError(e.loc, `this is undefined`)
        return this.block.readVar(this.thisVar)
      case 'ArrayExpression':
        return this.unhandledExpression(e)
      case 'ObjectExpression':
        return this.translateObjectExpression(e)
      case 'FunctionExpression':
        {
          const sym = this.module.freshFunctionId(null)
          const fun = new FunctionContext(sym)
          const ts = new Scope(this, { newFunction: fun })
          ts.translateFunctionExpression(e)
          const r = this.block.freshValue()
          this.block.appendStatement(new StdConstant(r, sym))
          return r
        }
      case 'ArrowFunctionExpression':
        return this.unhandledExpression(e)
      case 'YieldExpression':
        return this.unhandledExpression(e)
      case 'Literal':
        return this.translateLiteral(e)
      case 'UnaryExpression':
        return this.translateUnaryExpression(e)
      case 'UpdateExpression':
        return this.unhandledExpression(e)
      case 'BinaryExpression':
        return this.translateBinaryExpression(e)
      case 'AssignmentExpression':
        return this.translateAssignmentExpression(e)
      case 'LogicalExpression':
        return this.translateLogicalExpression(e)
      case 'MemberExpression':
        return this.translateMemberExpression(e)
      case 'ConditionalExpression':
        return this.translateConditionalExpression(e)
      case 'CallExpression':
        return this.translateCallExpression(e)
      case 'NewExpression':
        return this.translateNewExpression(e)
      case 'SequenceExpression':
        return this.unhandledExpression(e)
      case 'TemplateLiteral':
        return this.translateTemplateLiteral(e)
      case 'TaggedTemplateExpression':
      case 'ClassExpression':
      case 'MetaProperty':
        return this.unhandledExpression(e)
      case 'Identifier':
        {
          LexicalEnvironment.

        return this.block.readVar(this.lookupVar(e))
      case 'AwaitExpression':
      case 'ImportExpression':
      case 'ChainExpression':
        return this.unhandledExpression(e)
      default:
        return assertNever(e)
      }
  }

  private useStrict: boolean = false

  unhandledStatement(e: estree.Statement) {
    this.addWarning(e.loc, `Unhandled statement: ${e.type}`)
  }

  /**
   * Translate a block statement
   * @param b Block statement
   * @returns true if we should keep reading
   */
  translateBlockStatement(b:estree.BlockStatement):boolean {
    for (const s of b.body) {
      const b:boolean = this.translateStatement(s)
      if (!b)
        return false
    }
    return true
  }

  translateIfStatement(s:estree.IfStatement) {
    this.translateExpression(s.test)
    this.translateStatement(s.consequent)
    if (s.alternate) {
      this.translateStatement(s.alternate)
    }
  }

  finalizeRegion():Region {
    return {} // FIXME
  }

  endBlock(term:mlir.TerminalOp) {
    const b = this.block
    this.function.addBlock({
      id: b.id,
      readVars: b.readVars,
      stmts: b.statements,
      term: term,
      assignedVars: b.assignedVars
    })
  }


  startFreshBlock(term:mlir.TerminalOp, nextBlock:BlockId) {
    this.endBlock(term)
    this.block = new BlockContext(nextBlock)
  }

  translateSwitchStatement(s:estree.SwitchStatement) {
    this.translateExpression(s.discriminant)
    const cases = new Array<Case>()
    const tests = new Set<CaseTest>()
    var defaultCase : BlockId|undefined = undefined
    // Block to use for after switch
    const nextBlock = this.function.freshBlockId()

    for (const c of s.cases) {
      const test = c.test

      const tl = new Scope(this, {breakBlock: nextBlock })
      const caseId = tl.block.id
      var terminated = false
      for (const st of c.consequent) {
        if (!tl.translateStatement(st)) {
          terminated = true
          break
        }
      }

      if (!terminated)
        tl.endBlock(new cf.BranchOp(nextBlock))

      if (test) {
        switch (test.type) {
        case 'Literal':
          if ('regex' in test) {
            this.addWarning(s.loc, `Unhandled regex switch case: ${test.type}`)
          } else if ('bigint' in test) {
            this.addWarning(s.loc, `Unhandled bigint switch case: ${test.type}`)
          } else {
            if (!test.value)
              throw new TranslationError(test.loc, 'Missing test value')
            if (tests.has(test.value))
              throw new TranslationError(test.loc, 'Duplicate test ${test.value}')
            tests.add(test.value)
            cases.push(new Case(test.value, caseId))
          }
          break;
        default:
          this.addWarning(c.loc, `Unhandled switch case: ${test.type}`)
        }
      } else {
        if (defaultCase)
          this.addWarning(c.loc, `Duplicate default cases`)
        defaultCase = caseId
      }
    }
    this.endBlock(new SwitchStmt(cases, defaultCase ? defaultCase : nextBlock))
    this.block = new BlockContext(nextBlock)
  }

  translateFunctionDeclaration(d:estree.FunctionDeclaration) {
    const id : estree.Identifier | null = d.id

    if (d.generator) {
      this.addWarning(d.loc, `Generator functions are not yet supported.`)
    }
    if (d.async) {
      this.addWarning(d.loc, `Async functions are not yet supported`)
    }

    const funId = this.module.freshFunctionId(d.id?d.id.name:null)
    const tl = new Scope(this, {newFunction: new FunctionContext(funId) });
    for (const p of d.params) {
      tl.addPatternIdent(p) // Value
    }
    tl.translateBlockStatement(d.body)

    if (id) {
      if (this.identMap.has(id.name))
        throw new TranslationError(id.loc, `${id.name} already defined.`)
      const synVar = this.function.freshVar(id.name)
      this.identMap.set(id.name, synVar)

//      this.block.assign(this.declareVar(id), value)

  //    this.addIdent(name.name, {})
    }
  }

  predefinedVar(name:string): FunctionVar {
    const synVar = this.function.freshVar(name)
    this.identMap.set(name, synVar)
    return synVar
  }


  declareVar(id:estree.Identifier): FunctionVar {
    if (this.identMap.has(id.name))
      throw new TranslationError(id.loc, `${id.name} already defined.`)
    return this.predefinedVar(id.name)
  }

  translateVariableDeclaration(d:estree.VariableDeclaration) {
    for (const decl of d.declarations) {
      const v = decl.init
      const value = v ? this.translateExpression(v) : this.block.undefinedJavascriptValue()
      const id = decl.id
      switch (id.type) {
        case 'Identifier':
          this.block.assign(this.declareVar(id), value)
          break;
        case 'ObjectPattern':
        case 'ArrayPattern':
        case 'RestElement':
        case 'AssignmentPattern':
        case 'MemberExpression':
          this.addWarning(id.loc, `Parameter ${id.type} unsupported`)
          break
      }
    }
  }

  /*
  interface FunctionInfo {
    scope: Scope
    symbol: mlir.Symbol
    superClass : Value|null
  }
  */


  translateClassDeclaration(d:estree.ClassDeclaration) {
    var superClass : Value|null
      = d.superClass
      ? this.translateExpression(d.superClass)
      : null

    const body : estree.ClassBody = d.body
    const clName = this.module.uniqueClassName(d.id)

    const ctor = this.module.freshFunctionId(clName)

    for (const b of body.body) {
      switch (b.type) {
        case 'MethodDefinition':
          switch (b.kind) {
            case 'constructor':
              assert(!b.static)
              assert(!b.computed)
              assert(b.key.type === 'Identifier' && b.key.name === 'constructor')

              const fun = new FunctionContext(ctor)
              const ts = new Scope(this, { newFunction: fun, superMethod: superClass })
              ts.thisVar = fun.freshVar('this')
              ts.translateFunctionExpression(b.value)
              break
            case 'method':
              {
                assert(!b.computed)
                if (b.key.type !== 'Identifier') {
                  throw new TranslationError(b.loc, `Expected method name to be identifier`)
                }
                const methodRef = this.module.freshFunctionId(`${ctor.name}_${b.key.name}`)
                const fun = new FunctionContext(methodRef)
                const ts = new Scope(this, { newFunction: fun })
                ts.thisVar = b.static ? null : fun.freshVar('this')
                ts.translateFunctionExpression(b.value)
              }
              break
            case 'get':
              throw new TranslationError(b.loc, `get unsupported ${JSON.stringify(b.key)}`)
            case 'set':
              throw new TranslationError(b.loc, `set unsupported ${JSON.stringify(b.key)}`)
            default:
              assertNever(b.kind)
          }
          break;
        case 'PropertyDefinition':
          {
            if (b.value) {
              this.translateExpression(b.value)
            }

            const key : estree.Expression | estree.PrivateIdentifier = b.key
            if (key.type === "PrivateIdentifier") {
              let name : string = key.name
            } else {
              this.translateExpression(key)
            }
            if (b.computed) {
              this.addWarning(b.loc, 'Unexpected computed property definition')
            }
            const isStatic : boolean = b.static
          }
          const pm : estree.PropertyDefinition = b
          break;
        case 'StaticBlock':
          throw new Error('Static block not yet supported.')
        default:
          assertNever(b)
      }
    }
  }

  translateBreakStatement(s:estree.BreakStatement) {
    if (!this.breakBlock)
      throw new TranslationError(s.loc, `'break' appears out of scope.`)
    this.endBlock(new cf.BranchOp(this.breakBlock))
  }

  translateContinueStatement(s:estree.ContinueStatement) {
    if (!this.continueBlock)
      throw new TranslationError(s.loc, `'continue' appears out of scope.`)
    this.endBlock(new cf.BranchOp(this.continueBlock))
  }

  // Translate return statement
  translateReturnStatement(s:estree.ReturnStatement) {
    const v = s.argument ? [{value : this.translateExpression(s.argument), type: 'FIXME'}] : []
    this.endBlock(new ReturnOp(v))
  }

  translateTryStatement(s:estree.TryStatement):boolean {
    const blockScope:Scope = new Scope(this, {})
    var c = blockScope.translateBlockStatement(s.block)

    if (s.handler) {
      const h = s.handler
      const handlerScope = new Scope(this, {})
      if (h.param)
        handlerScope.addPatternIdent(h.param)

      const hContinue = handlerScope.translateBlockStatement(h.body)
      if (hContinue)
        c = true
    }
    if (s.finalizer) {
      const finalizerScope = new Scope(this, {})
      finalizerScope.translateBlockStatement(s.finalizer)
    }
    return c
  }

  checkTopLevel(loc:estree.SourceLocation|undefined|null, msg: string):any {
    if (this.parent != null)
      throw new TranslationError(loc,  msg)
  }

  translateThrowStatement(s: estree.ThrowStatement) {
    const a = this.translateExpression(s.argument)
    if (this)
    this.endBlock(new ThrowOp(a))
  }

  // End the current block with a jump to a fresh block id
  // Return id of this block
  endBlockWithJumpToFresh():BlockId {
    const blockId = this.function.freshBlockId()
    this.startFreshBlock(new cf.BranchOp(blockId), blockId)
    return blockId
  }

  private translateWhileStatement(s: estree.WhileStatement) {


    const testId = this.endBlockWithJumpToFresh()
    const v = this.translateExpression(s.test)
    const nextId = this.function.freshBlockId()

    this.startFreshBlock(new cf.CondBranchOp(v, testId, nextId), nextId)

    let scope = new Scope(this, {breakBlock: nextId, continueBlock: testId})
    const b = scope.translateStatement(s.body)
    if (b) {
      this.endBlock(new cf.BranchOp(testId))
    }


  }

  translateDoWhileStatement(s: estree.DoWhileStatement) {

  }

  translateForStatement(s: estree.ForStatement) {

  }

  translateForInStatement(s: estree.ForInStatement) {

  }

  translateForOfStatement(s: estree.ForOfStatement) {

  }

  /**
   * Process declarations and ensure they are added to environment.
   * @param s Declaration to add.
   */
  processDeclarations(s: estree.Directive| estree.Statement | estree.ModuleDeclaration) {
    switch (s.type) {
      case 'ClassDeclaration':
      {
        const id : estree.Identifier | null = s.id
        if (id) {
          const synVar = this.function.freshVar(id ? id.name : undefined)
          if (this.identMap.has(id.name))
            throw new TranslationError(id.loc, `Duplicate identifier ${id.name}.`)
          this.identMap.set(id.name, synVar)
        }
        break
      }
      default:
        break
    }
  }

  /**
   * Translate a directive, statement or module declaration by appending to current block.
   * Directives and module declarations are only allowed at the top-level (i.e. when parent = null)
   *
   * If false is returned then it is the responsibility of translateStatement to
   * add current statements to a block.
   *
   * @param s estree node to translate
   * @returns true if we should continue with additional statements
   * and false if control flow ends (e.g., with return, break or throw).
   */
  translateStatement(s: estree.Directive| estree.Statement | estree.ModuleDeclaration):boolean {
    switch (s.type) {
      // Statement or directive
      case 'ExpressionStatement':
        if ('directive' in s) {
          this.checkTopLevel(s.loc, `Directive expected only at top level.`)
          switch (s.directive) {
            case 'use strict':
              this.useStrict = true
              break
            default:
              this.addWarning(s.loc, `Unknown directive: ${s.directive}`)
              break;
          }
          return true
        } else {
          this.translateExpression(s.expression)
          return true
        }
      case 'BlockStatement':
        {
          const tl = new Scope(this, {})
          const c = tl.translateBlockStatement(s)
          return c
        }
      case 'EmptyStatement':
        // Do nothing
        return true
      case 'DebuggerStatement':
        this.unhandledStatement(s)
        return true
      case 'WithStatement':
        this.unhandledStatement(s)
        return true
      case 'ReturnStatement':
        this.translateReturnStatement(s)
        return false
      case 'LabeledStatement':
        this.unhandledStatement(s)
        return true
      case 'BreakStatement':
        this.translateBreakStatement(s)
        return false
      case 'ContinueStatement':
        this.translateContinueStatement(s)
        return false
      case 'IfStatement':
        this.translateIfStatement(s)
        return true
      case 'SwitchStatement':
        this.translateSwitchStatement(s)
        return true
      case 'ThrowStatement':
        this.translateThrowStatement(s)
        return false
      case 'TryStatement':
        return this.translateTryStatement(s)
      case 'WhileStatement':
        this.translateWhileStatement(s)
        return true
      case 'DoWhileStatement':
        this.translateDoWhileStatement(s)
        return true
      case 'ForStatement':
        this.translateForStatement(s)
        return true
      case 'ForInStatement':
        this.translateForInStatement(s)
        return true
      case 'ForOfStatement':
        this.translateForOfStatement(s)
        return true
      // Declaration
      case 'FunctionDeclaration':
        this.translateFunctionDeclaration(s)
        return true
      case 'VariableDeclaration':
        this.translateVariableDeclaration(s)
        return true
      case 'ClassDeclaration':
        this.translateClassDeclaration(s)
        return true
      // Module declaration
      case 'ImportDeclaration':
        this.checkTopLevel(s.loc, `Import expected only at top level.`)
        this.addWarning(s.loc, `Unhandled ${s.type}`)
        return true
      case 'ExportDefaultDeclaration':
        this.checkTopLevel(s.loc, `Export default expected only at top level.`)
        this.addWarning(s.loc, `Unhandled ${s.type}`)
        return true
      case 'ExportNamedDeclaration':
        this.checkTopLevel(s.loc, `Export expected only at top level.`)
        this.addWarning(s.loc, `Unhandled ${s.type}`)
        return true
      case 'ExportAllDeclaration':
        this.checkTopLevel(s.loc, `Exportall expected only at top level.`)
        this.addWarning(s.loc, `Unhandled ${s.type}`)
        return true
      default:
        return assertNever(s)
    }
  }
}


if (process.argv.length != 5) {
  process.stderr.write('Please provide (script|module) javascript_filename mlir_path.\n')
  process.exit(-1)
}
// Get filetype
const fileType = process.argv[2];

// Get path of Javascript file to open
const inputPath = process.argv[3];

// Get path of MLIR file to write
const outputPath = process.argv[4];

var contents : string;
try {
  contents = fs.readFileSync(inputPath).toString()
} catch (error) {
  process.stderr.write(`Could not read ${inputPath}.\n`)
  process.exit(-1)
}

try {
  var s : esprima.Program;
  const config : esprima.ParseOptions = {loc: true}
  switch (fileType) {
    case 'module':
      s = esprima.parseModule(contents, config)
      break;
    case 'script':
      s = esprima.parseScript(contents, config)
      break;
    default:
      process.stderr.write('Please specify whether script or module.\n')
      process.exit(-1)
  }

  const tl = new Scope(null, {})
  tl.predefinedVar('process')
  tl.predefinedVar('undefined')
  tl.predefinedVar('exports')
  tl.predefinedVar('require')
  tl.predefinedVar('console')

  tl.predefinedVar('Array')
  tl.predefinedVar('Error')
  tl.predefinedVar('JSON')
  tl.predefinedVar('Map')
  tl.predefinedVar('Object')
  tl.predefinedVar('Set')
  s.body.forEach(d => tl.processDeclarations(d))
  for (const d of s.body) {
    tl.translateStatement(d)
  }
  for (const w of tl.warnings) {
    process.stderr.write(`${formatWarning(w)}\n`)
  }
  const mlirStream = fs.createWriteStream(outputPath);
  for (const op of tl.module.ops) {
    op.write(mlirStream, '')
  }
  mlirStream.close()

} catch (error) {
  process.stderr.write(`Error: ${error.message}\n`)
  process.stderr.write(`Stack: ${error.stack}\n`)

  process.exit(1)
}