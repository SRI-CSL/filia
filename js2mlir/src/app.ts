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
import { BlockId, Block, BlockArgDecl, Op, Value } from "./mlir"

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

class MlirModule {
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

/*
// Creates a map from blocks to their index.
function mkBlockIdMap(blocks:BlockInfo[]):Map<BlockId, number> {
  // Map from blocks ids to index in blocks[]
  const blockMap: Map<BlockId, number> = new Map()
  for (var i=0; i < blocks.length; ++i) {
    blockMap.set(blocks[i].id, i)
  }
  return blockMap
}
*/

/*
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
*/

/*
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
*/

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

/*
interface BlockInfo {
  id: BlockId,
  inputs: BlockArgDecl[],
  stmts:Op[],
  term:mlir.TerminalOp,
  assignedVars : AssignedVars
}
*/

/*
function blockReadClosure(blocks:BlockInfo[], succMap: number[][]): Array<Set<BlockInput>> {

    // Compute what variables each block needs.
    var changed:boolean
    // Maps from block ids to the function variables that they need.
    const blockInputVarSet : Array<Set<BlockInput>> = new Array(blocks.length)

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
*/

// Information about a MLIR function being constructed by the translator.
class FunctionContext {
  // Symbol identifying function
  symbol: mlir.Symbol

  // Blocks to add  to the current function
  private readonly blocks:Array<Block> = new Array()

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
  freshValue(name?:js.Ident):Value {
    const gen = this.gen
    return name ? gen.freshIdent(name) : gen.freshAnon()
  }

  // Add block to end of blocks
  addBlock(b: Block) {
    this.blocks.push(b)
  }

  getFunction() : mlirb.Func {
    process.stderr.write(`Function ${this.blocks.length}\n`)
    const argTypes : mlirb.FunArg[] = []
    const region = { blocks: this.blocks }
    const retType = mlirjs.ValueType
    return new mlirb.Func(this.symbol, mlirb.funType(argTypes, retType), region)
  }

}

/// Information about the current block being generated.
class BlockContext {
  // Identifier for this block
  readonly id:BlockId

  // Variables read from previous block needed to compute block.
  readonly inputs : Array<BlockArgDecl> = new Array()

  // Statements for the current block.
  statements:Array<Op>|null

  // Local argument to block
  localsArg: BlockArgDecl

  // Value assigned to local variables
  locals: Value

  constructor(readonly fun:FunctionContext, blockId: BlockId, entryBlock?: boolean) {
    this.id = blockId
    this.statements = new Array()
    this.locals = fun.freshValue("locals")
    if (entryBlock) {
      this.statements.push(new mlirjs.EmptyLocals(this.locals))
    } else {
      this.inputs.push({ name: this.locals, type: mlirjs.LocalsType })
    }
  }

  // Create a new block input
  addBlockInput(type: mlir.TypeAttr, name? : js.Ident):BlockArgDecl {
    const r = { name: this.fun.freshValue(name), type: type }
    this.inputs.push(r)
    return r
  }

  private appendStatement(op: Op) {
    if (!this.statements)
      throw new Error('Block already complete.')
    if (op.isTerminal())
      throw new Error('Internal statement is terminal.')
    this.statements.push(op)
  }

  /** Return a value equal to null. */
  number(x:number):Value {
    let v = this.fun.freshValue()
    this.appendStatement(new mlirjs.Number(v, x))
    return v
  }

  /** Return a value equal to null. */
  null():Value {
    let v = this.fun.freshValue()
    this.appendStatement(new mlirjs.Null(v))
    return v
  }

  /** Call a given function */
  call(callee: Value, args: mlirjs.CallArg[], optional: boolean):Value {
    const r = this.fun.freshValue()
    this.appendStatement(new mlirjs.CallOp(r, callee, args, optional))
    return r
  }

  /** Return a value equal to undefined. */
  undefinedValue():Value {
    const v = this.fun.freshValue()
    this.appendStatement(new mlirjs.Undefined(v))
    return v
  }

  templateLiteral(quasis:string[], args: Value[]): Value {
    const r = this.fun.freshValue()
    this.appendStatement(new mlirjs.TemplateLiteral(r, quasis, args))
    return r
  }

  /** Return i1 value indicating if we are truthy. */
  truthy(x:Value):Value {
    let r = this.fun.freshValue()
    this.appendStatement(new mlirjs.Truthy(r, x))
    return r
  }

  getProperty(obj: Value, name: string): Value {
    let r = this.fun.freshValue()
    this.appendStatement(new mlirjs.GetProperty(r, obj, name))
    return r
  }

  stringLit(value: string): Value {
    let r = this.fun.freshValue()
    this.appendStatement(new mlirjs.StringLit(r, value))
    return r
  }

  /** Declare a variable and set initial value. */
  declVariable(kind: mlirjs.VarKind, name: string, value:Value|undefined) {
    const r = this.fun.freshValue("locals")
    this.appendStatement(new mlirjs.LocalDecl(r, this.locals, kind, name, value))
    this.locals = r
  }

  /** Set variable lhs to value and update environment. */
  setVariable(lhs: string, value: Value) {
    const r = this.fun.freshValue("locals")
    this.appendStatement(new mlirjs.LocalSet(r, this.locals, lhs, value))
    this.locals = r
  }

  /** Get variable with name from context. */
  getVariable(v: string): Value {
    const r = this.fun.freshValue(v)
    this.appendStatement(new mlirjs.LocalGet(r, this.locals, v))
    return r
  }


  endBlock(term:mlir.TerminalOp) {
    if (!this.statements)
      throw new Error('Block already ended')
    this.statements.push(term)
    this.fun.addBlock(new Block(this.id, this.inputs, this.statements))
    this.statements = null
  }

  branch(id:BlockId) {
    const args = [{ value: this.locals, type: mlirjs.LocalsType }]
    this.endBlock(new cf.BranchOp({id: id, args: args}))
  }

  condBranch(cond: Value, trueId: BlockId, falseId: BlockId) {
    const args = [{ value: this.locals, type: mlirjs.LocalsType }]
    this.endBlock(new cf.CondBranchOp(cond, {id: trueId, args: args}, {id: falseId, args: args}))
  }

  return(val:Value|null) {
    const v = val
      ? [{value : val, type: mlirjs.ValueType}]
      : []
    this.endBlock(new ReturnOp(v))
  }
}



interface ScopeConfig {
  // New value to use for this (or undefined to copy from parent).
  // Method to use for invoking super
  superMethod?:Value|null
  // Module to use
  newModule?:MlirModule
  // Start a fresh function (rather than use parent)
  newFunction?:FunctionContext
  // Append statements to parent block (default to create new block)
  newBlock?:BlockId
  // Value for this (or undefined to inherit from parent scope).
  //newThis?:BlockInput|null
  continueBlock?:BlockId|null
  breakBlock?:BlockId|null
}


class Scope {
  // Global warnings array
  readonly warnings: Array<Warning>;

  // Parent for lookup scope
  private readonly parent: Scope|null

  // Length of parent chain (used for building references)
  private readonly level: number

  module: MlirModule

  // Information about function
  private function: FunctionContext

  // Information from current block.
  block: BlockContext

  // Set of identifiers defined in this scope
//  private readonly identSet: Set<string>

  // Value to use when translating "this"
  //thisVar: BlockInput|null

  // Value that denotes function to invoke with super(...)
  private readonly superMethod:Value|null

  // Block to jump to for continue or null if a continue should not be allowed
  private continueBlock:BlockId|null

  // Block to jump to for break or null if a break should not be allowed
  private breakBlock:BlockId|null

  constructor(parent:Scope|null, config:ScopeConfig) {
    this.warnings = parent ? parent.warnings : new Array<Warning>()
    this.parent = parent
    this.level = parent ? parent.level + 1 : 0
    if (config.newModule) {
      this.module = config.newModule
    } else {
      if (!parent)
        throw new Error('Expected parent when newModule not set')
      this.module = parent.module
    }

    if (config.newFunction) {
      this.function = config.newFunction
      const blockId = this.function.freshBlockId()
      this.block = new BlockContext(this.function, blockId, true)
    } else {
      if (!parent) {
        throw new Error('Expected parent when newFunction not set')
      }
      this.function = parent.function
      if (config.newBlock) {
        this.block = new BlockContext(this.function, config.newBlock)
      } else {
        this.block = parent.block
      }
    }

    /*
    if (config.newThis !== undefined) {
      this.thisVar = config.newThis
    } else {
      if (!parent)
        throw new Error('Expected parent when newThis is not set')
      this.thisVar = parent.thisVar
    }
    */
    this.superMethod
      = config.superMethod !== undefined ? config.superMethod
      : parent ? parent.superMethod
      : null
    this.continueBlock
      = config.continueBlock !== undefined ? config.continueBlock
      : parent ? parent.continueBlock
      : null
    this.breakBlock
      = config.breakBlock !== undefined ? config.breakBlock
      : parent ? parent.breakBlock
      : null
  }

  addWarning(loc:estree.SourceLocation|null|undefined, msg: string) {
    this.warnings.push({loc: loc, message: msg})
  }

  unhandledExpression(e: estree.Expression):Value {
    this.addWarning(e.loc, `Unhandled expression: ${e.type}`)
    return this.block.undefinedValue()
  }

  unhandledPattern(p:estree.Pattern) {
    this.addWarning(p.loc,`Unhandled pattern: ${p.type}`)
  }

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

    const c = this.translateStatements(e.body.body)
    if (c)
      this.block.return(null)

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
        return this.block.undefinedValue()
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
        return this.block.undefinedValue()
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
        this.block.setVariable(left.name, v)
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
        return this.block.undefinedValue()
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
        return this.block.getProperty(o, name)
      default:
        this.addWarning(e.loc, `property ${e.property.type} unsupported`)
        return this.block.undefinedValue()
    }
  }

  private translateConditionalExpression(e: estree.ConditionalExpression): Value {
    const c = this.block.truthy(this.translateExpression(e.test))
    const trueId = this.function.freshBlockId()
    const falseId = this.function.freshBlockId()
    const nextBlockId = this.function.freshBlockId()

    this.block.condBranch(c, trueId, falseId)


    const trueScope = new Scope(this, {newBlock: trueId})
    const trueValue = trueScope.translateExpression(e.consequent)
    const trueBlock = trueScope.block
    const trueArgs = [
      { value: trueBlock.locals, type: mlirjs.LocalsType },
      { value: trueValue, type: mlirjs.ValueType }
    ]
    trueBlock.endBlock(new cf.BranchOp({id: nextBlockId, args: trueArgs}))

    const falseScope = new Scope(this, {newBlock: falseId})
    const falseValue = falseScope.translateExpression(e.alternate)
    const falseBlock = falseScope.block
    const falseArgs = [
      { value: falseBlock.locals, type: mlirjs.LocalsType },
      { value: falseValue, type: mlirjs.ValueType }
    ]
    falseBlock.endBlock(new cf.BranchOp({id: nextBlockId, args: falseArgs}))

    this.block =  new BlockContext(this.function, nextBlockId)
    const r = this.block.addBlockInput(mlirjs.ValueType)
    return r.name
  }

  /*
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
  */

  translateCallExpression(e: estree.SimpleCallExpression):Value {

    if (e.callee.type === 'Super') {
      this.addWarning(e.callee.loc, `Calling super unsupported`)
      return this.block.undefinedValue()
    }
    const callee = this.translateExpression(e.callee)
    const args:mlirjs.CallArg[] = []
    for (const a of e.arguments) {
      let e:estree.Expression
      let spread:boolean
      if (a.type === 'SpreadElement') {
        e = a.argument
        spread = true
      } else {
        e = a
        spread = false
      }
      args.push({value: this.translateExpression(e), spread: spread})
    }
    // Indicates that if callee is null or undefined then we should skip
    const optional:boolean = e.optional

    return this.block.call(callee, args, optional)
  }

  translateNewExpression(e: estree.NewExpression):Value {
    this.addWarning(e.loc, `new expression unsupported`)
    return this.block.undefinedValue()
  }

  translateIdentifier(e: estree.Identifier): Value {
    return this.block.getVariable(e.name)
  }

  /**
   * Translate a template literal (e.g., `Hello ${var}`)
   */
  translateTemplateLiteral(l: estree.TemplateLiteral): Value {
    const quasis = l.quasis.map((e) => e.value.raw)
    const args = l.expressions.map((e) => this.translateExpression(e))

    return this.block.templateLiteral(quasis, args)
  }


  translateObjectExpression(e:estree.ObjectExpression): Value {
    this.addWarning(e.loc, `object expression unsupported`)
    return this.block.undefinedValue()
    /*
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
    */
  }

  unhandledLiteral(s: string, e: estree.Literal):Value {
    this.addWarning(e.loc, `Unhandled literal: ${s}`)
    return this.block.undefinedValue()
  }

  translateLiteral(e: estree.Literal):Value {
    switch (typeof e.value) {
      case 'bigint':
        return this.unhandledLiteral('bigint', e)
      case 'boolean':
        return this.unhandledLiteral('boolean', e)
      case 'number':
        return this.block.number(e.value)
      case 'object':
        if (e.value instanceof RegExp) {
          return this.unhandledLiteral('RegExp', e)
        } else if (e.value === null) {
          return this.block.null()
        } else {
          return assertNever(e.value)
        }
      case 'string':
        return this.block.stringLit(e.value)
      case 'undefined':
        return this.block.undefinedValue()
      default:
        return assertNever(e.value)
    }
  }


  translateExpression(e: estree.Expression):Value {
    process.stderr.write(`$Translate expr ${e.type}\n`)
    switch (e.type) {
      case 'ThisExpression':
        return this.unhandledExpression(e)
      case 'ArrayExpression':
        return this.unhandledExpression(e)
      case 'ObjectExpression':
        return this.translateObjectExpression(e)
      case 'FunctionExpression':
        return this.unhandledExpression(e)
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
        return this.translateIdentifier(e)
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
  translateStatements(stmts:(estree.Directive|estree.Statement|estree.ModuleDeclaration)[]):boolean {
    for (const s of stmts) {
      const c = this.translateStatement(s)
      if (!c)
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

  translateSwitchStatement(s:estree.SwitchStatement) {
    this.unhandledStatement(s)
    /*
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

      if (!terminated) {
        tl.block.branch(nextBlock)
      }
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
    this.block.endBlock(new SwitchStmt(cases, defaultCase ? defaultCase : nextBlock))
    this.block = new BlockContext(this.function, nextBlock)
    */
  }

  translateVariableDeclaration(d:estree.VariableDeclaration) {
    for (const decl of d.declarations) {
      const v = decl.init
      const value : Value|undefined = v ? this.translateExpression(v) : undefined
      const id = decl.id
      switch (id.type) {
        case 'Identifier':
          this.block.declVariable(d.kind, id.name, value)
          break;
        case 'ObjectPattern':
        case 'ArrayPattern':
        case 'RestElement':
        case 'AssignmentPattern':
        case 'MemberExpression':
          this.addWarning(id.loc, `Variable declaration ${id.type} unsupported`)
          break
      }
    }
  }

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

              const fun = new FunctionContext(ctor)
              const ts = new Scope(this, { newFunction: fun, superMethod: superClass })
              //ts.thisVar = { varId: fun.freshValue('this') }
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
                //ts.thisVar = b.static ? null : { varId: fun.freshValue('this') }
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
    this.block.branch(this.breakBlock)
  }

  translateContinueStatement(s:estree.ContinueStatement) {
    if (!this.continueBlock)
      throw new TranslationError(s.loc, `'continue' appears out of scope.`)
    this.block.branch(this.continueBlock)
  }

  // Translate return statement
  translateReturnStatement(s:estree.ReturnStatement) {
    const v = s.argument
      ? this.translateExpression(s.argument)
      : null
    this.block.return(v)
  }

  translateTryStatement(s:estree.TryStatement):boolean {
    this.unhandledStatement(s)
    return true
    /*
    const blockScope:Scope = new Scope(this, {})
    var c = blockScope.translateStatements(s.block.body)

    if (s.handler) {
      this.addWarning(s.handler.loc, `Try exception handler unsupported.`)
      c = true
    }
    if (s.finalizer) {
      this.addWarning(s.finalizer.loc, `Try finalizer unsupported.`)
      c = true
    }
    return c
    */
  }

  checkTopLevel(loc:estree.SourceLocation|undefined|null, msg: string):any {
    if (this.parent != null)
      throw new TranslationError(loc,  msg)
  }

  private translateWhileStatement(s: estree.WhileStatement) {

    const testId = this.function.freshBlockId()
    const bodyId = this.function.freshBlockId()
    const nextId = this.function.freshBlockId()

    this.block.branch(testId)

    const testBlock = new BlockContext(this.function, testId)
    this.block = testBlock
    const c = testBlock.truthy(this.translateExpression(s.test))
    testBlock.condBranch(c, bodyId, nextId)

    let scope = new Scope(this, {newBlock: bodyId, breakBlock: nextId, continueBlock: testId})
    const b = scope.translateStatement(s.body)
    if (b) {
      scope.block.branch(testId)
    }

    this.block = new BlockContext(this.function, nextId)
  }

  translateDoWhileStatement(s: estree.DoWhileStatement) {
    this.addWarning(s.loc, `Do while unsupported.`)
  }

  translateForStatement(s: estree.ForStatement) {
    this.addWarning(s.loc, `For unsupported.`)
  }

  translateForInStatement(s: estree.ForInStatement) {
    this.addWarning(s.loc, `For in unsupported.`)
  }

  translateForOfStatement(s: estree.ForOfStatement) {
    this.addWarning(s.loc, `For of unsupported.`)
  }

  /**
   * Process declarations and ensure they are added to environment.
   * @param s Declaration to add.
   */
/*
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
  */

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
          const c = tl.translateStatements(s.body)
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
        this.unhandledStatement(s)
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
        this.unhandledStatement(s)
        return true
      case 'StaticBlock':
        this.unhandledStatement(s)
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
  const module = new MlirModule()
  const main = new FunctionContext(module.freshFunctionId('script_main'))

  const tl = new Scope(null, {newModule: module, newFunction: main})
  const c = tl.translateStatements(s.body)
  if (c)
    tl.block.return(null)

  module.ops.push(main.getFunction())
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