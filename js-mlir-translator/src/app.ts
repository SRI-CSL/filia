import { assert } from 'console'
import * as esprima from 'esprima'
import * as estree from 'estree'
import * as fs from 'fs'
import { runInThisContext } from 'vm'
import * as js from './javascript'
import * as mlir from "./mlir"
import * as mlirjs from './mlir/javascript'
import { BlockId, Block, Op, mlir.Symbol, Value } from "./mlir"

if (process.argv.length != 4) {
  process.stderr.write('Please provide filename and script or module.\n')
  process.exit(-1)
}
// Get filetype
const fileType = process.argv[2];

// Get path of Javascript file to open
const path = process.argv[3];

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

class SwitchStmt implements mlir.Op {
  constructor(cases:Case[], defaultCase:BlockId) {

  }

  write(s: NodeJS.WriteStream) {
      // FIXME
  }
}

class Jump implements mlir.Op {
  constructor(target:BlockId) {

  }
  write(s: NodeJS.WriteStream) {
    // FIXME
  }
}

class Branch implements mlir.Op {
  constructor(test:mlir.Value, trueTarget:BlockId, falseTarget:BlockId) {

  }

  write(s: NodeJS.WriteStream) {
    // FIXME
  }
}

class StdConstant implements mlir.Op {
  constructor(ret:mlir.Value, sym:mlir.Symbol) {
    // FIXME

  }
  write(s: NodeJS.WriteStream) {
    // FIXME
  }
}



class ReturnOp implements mlir.Op {
  constructor(ret:mlir.Value|null) {

  }
  write(s: NodeJS.WriteStream) {
    // FIXME
  }
}

class ThrowOp implements mlir.Op {
  constructor(v:mlir.Value) {

  }
  write(s: NodeJS.WriteStream) {
    // FIXME
  }
}

interface ClassInfo {
  readonly classConstructor:mlir.Symbol
}





class Module {
  private gen = new IdentGenerator('@')

  freshFunctionId(x: estree.Identifier|null|undefined):mlir.Symbol {
    let val = x ? this.gen.freshIdent(x.name) : this.gen.freshAnon()
    return mlir.mkSymbol(val)
  }

  addFunction(fun:FunctionContext) {
    // FIXME
  }
}

class IdentGenerator {
  prefix: string

  // Maps identifiers to number of times it has been used so we
  // can generate fresh numbers.
  private identIndexMap: Map<js.Ident, number>

  // Generator for values not formed from identifiers.
  private nextValueIndex: number

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

// A synthetic variable local to a function
interface SyntheticVar {
  type: "var"
  varId: string
}

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
    return res
  }

  // Create a new syntheticv variable scope to function
  freshSyntheticVar(name?:js.Ident):SyntheticVar {
    const gen = this.gen
    return { type: "var", varId: name ? gen.freshIdent(name) : gen.freshAnon() }
  }

  addBlock(ident:BlockId, stmts:Op[]) {
    // FIXME
  }

}

// Strings are used for Javascript identifiers while numbers are used for synthetic
// variables.
type Var = js.Ident|SyntheticVar


/// Information about the current block being generated.
class BlockContext {
  // Identifier for this block
  readonly id:BlockId

  // Statements for the current block.
  statements:Array<Op> = new Array()

  // List of variables that are needed to compute block.
  private readVars : Array<SyntheticVar> = new Array()

  // Map from variables to their value in the block.
  private assignedVars : Map<SyntheticVar, Value> = new Map()

  private valueCount : number = 0

  constructor(blockId: BlockId) {
    this.id = blockId
  }

  appendStatement(op: Op) {
    this.statements.push(op)
  }

  freshValue(): Value {
    return `%$${this.valueCount++}`
  }

  assign(lhs: SyntheticVar, rhs:Value) {
    this.assignedVars.set(lhs, rhs)
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
  readVar(v: SyntheticVar): Value {
    var val = this.assignedVars.get(v)
    if (val === undefined) {
      val = v.varId
      this.readVars.push(v)
      this.assignedVars.set(v, val)
    }
    return val
  }

  /** Return a value equal to undefined. */
  undefinedJavascriptValue():Value {
    let v = this.freshValue()
    this.appendStatement(new mlirjs.Undefined(v))
    return v
  }
}

/** A reference to a variable (which may be mutable) */
//type Reference = number

interface ScopeConfig {
  // New value to use for this (or undefined to copy from parent).
  inheritThis?:boolean
  // Method to use for invoking super
  superMethod?:Value|null
  // Start a fresh function (rather than use parent)
  newFunction?:FunctionContext
  // Append statements to parent block (default to create new block)
  appendStatementsToParent?:boolean
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

  module: Module

  // Information about function
  private function: FunctionContext

  // Information from current block.
  private block: BlockContext

  // Set of identifiers defined in this scope
//  private readonly identSet: Set<string>

  // Value to use when translating "this"
  thisValue:Value|null

  // Value that denotes function to invoke with super(...)
  private readonly superMethod:Value|null

  // Block to jump to for continue or null if a continue should not be allowed
  private continueBlock:BlockId|null

  // Block to jump to for break or null if a break should not be allowed
  private breakBlock:BlockId|null

  // Maps string to variables local to function or global symbols
  private readonly identMap: Map<string, SyntheticVar|mlir.Symbol> = new Map()

  // Map class names to information for class
  private readonly classMap: Map<string, ClassInfo>

  constructor(parent:Scope|null, config:ScopeConfig) {
    this.warnings = parent ? parent.warnings : new Array<Warning>()
    this.parent = parent
    this.level = parent ? parent.level + 1 : 0
    this.module = parent ? parent.module : new Module()
    if (config.newFunction) {
      this.function = config.newFunction
    } else if (parent) {
      this.function = parent.function
    } else {
      const nm = this.module.freshFunctionId(null)
      this.function = new FunctionContext(nm)
    }

    if (config.appendStatementsToParent) {
      if (!parent) {
        throw new Error('Expected parent when appendStatementToParent set')
      }
      this.block = parent.block
    } else {
      this.block = new BlockContext(this.function.freshBlockId())
    }

    this.thisValue
      = parent && (config.inheritThis !== false) ? parent.thisValue : null
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
    this.classMap = new Map()
  }

  addStatement(op:Op):void {
    this.block.appendStatement(op)
  }

  addClass(name:string, value:ClassInfo) {
    this.classMap.set(name, value)
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

  addPatternIdent(p:estree.Pattern):Value {
    switch (p.type) {
      case 'Identifier':
        {
          const v = this.function.freshSyntheticVar(p.name)
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


  translateFunctionExpression(e: estree.FunctionExpression) {
    const ident : estree.Identifier|null|undefined = e.id

    if (e.generator) {
      throw new TranslationError(e.loc, 'Generator functions are not yet supported.')
    }
    if (e.async) {
      throw new TranslationError(e.loc, 'Async functions are not yet supported.')
    }

    for (const p of e.params) {
      const val = this.addPatternIdent(p) // Value
    }

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

  lookupVar(v: estree.Identifier): SyntheticVar|mlir.Symbol {
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

  recordIdent(v: estree.Identifier, val:SyntheticVar) {

  }

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
        this.block.assign(this.lookupVar(left), v)
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

    console.log(`Member expression ${e.object.type}`)
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

    const v = this.function.freshSyntheticVar()

    const trueScope = new Scope(this, {})
    const trueId = trueScope.block.id

    const falseScope = new Scope(this, {})
    const falseId = falseScope.block.id

    const nextBlock = this.function.freshBlockId()

    const trueValue = trueScope.translateExpression(e.consequent)
    trueScope.block.assign(v, trueValue)
    trueScope.endBlock(new Jump(nextBlock))

    const falseValue = falseScope.translateExpression(e.alternate)
    falseScope.block.assign(v, falseValue)
    falseScope.endBlock(new Jump(nextBlock))

    this.startFreshBlock(new Branch(c, trueId, falseId), nextBlock)

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

  findClass(loc:estree.SourceLocation|null|undefined, name:string):ClassInfo {
    var cur:Scope|null = this
    while (cur) {
      const val = cur.classMap.get(name)
      if (val)
        return val
      cur = cur.parent
    }
    throw new TranslationError(loc, `Unknown class ${name}`)
  }

  translateNewExpression(e: estree.NewExpression):Value {
    var cl: ClassInfo
    switch (e.callee.type) {
      case 'Identifier':
        cl = this.findClass(e.callee.loc, e.callee.name)
        break
      default:
        throw new TranslationError(e.callee.loc, `Unsupported callee ${e.callee.type}`)
    }
    const argList = this.translateCallArguments(e.arguments)

    const callee = cl.classConstructor

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
        if (!this.thisValue)
          throw new TranslationError(e.loc, `this is undefined`)
        return this.thisValue
      case 'ArrayExpression':
        return this.unhandledExpression(e)
      case 'ObjectExpression':
        return this.translateObjectExpression(e)
      case 'FunctionExpression':
        {
          const sym = this.module.freshFunctionId(undefined)
          const fun = new FunctionContext(sym)
          const ts = new Scope(this, { newFunction: fun })
          ts.translateFunctionExpression(e)
          const r = this.block.freshValue()
          this.block.appendStatement(new StdConstant(r, sym))
          return r
        }
      case 'ArrowFunctionExpression':
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

  endBlock(s:mlir.Op) {
    const b = this.block
    b.appendStatement(s)
    this.function.addBlock(b.id, b.statements)
  }


  startFreshBlock(s:mlir.Op, nextBlock:BlockId) {
    const b = this.block
    b.appendStatement(s)
    this.function.addBlock(b.id, b.statements)
    this.block = new BlockContext(nextBlock)
  }

  translateSwitchStatement(s:estree.SwitchStatement) {
    this.translateExpression(s.discriminant)
    const cases = new Array<Case>()
    const tests = new Set<CaseTest>()
    var defaultCase : BlockId|undefined = undefined
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
        tl.endBlock(new Jump(nextBlock))

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
    this.startFreshBlock(new SwitchStmt(cases, defaultCase ? defaultCase : nextBlock), nextBlock)
  }

  translateFunctionDeclaration(d:estree.FunctionDeclaration) {
    const id : estree.Identifier | null = d.id

    this.addWarning(d.loc, `Function generator ${d.generator}`)
    if (d.async) {
      this.addWarning(d.loc, `Async functions not yet supported`)
    }

    const tl = new Scope(this, {});

    for (const p of d.params) {
      tl.addPatternIdent(p) // Value
    }
    tl.translateBlockStatement(d.body)

    /*
    if (id) {
      if (this.identMap.has(id.name))
        throw new TranslationError(id.loc, `${id.name} already defined.`)
      const synVar = this.function.freshSyntheticVar(id.name)
      this.identMap.set(id.name, synVar)

      this.block.assign(this.declareVar(name), value)

      this.addIdent(name.name, {})
    }
    */
  }

  declareVar(id:estree.Identifier): SyntheticVar {
    if (this.identMap.has(id.name))
      throw new TranslationError(id.loc, `${id.name} already defined.`)
    const synVar = this.function.freshSyntheticVar(id.name)
    this.identMap.set(id.name, synVar)
    return synVar
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

  translateClassDeclaration(d:estree.ClassDeclaration) {
    const id : estree.Identifier | null = d.id
//    const superClass : estree.Expression | null | undefined = d.superClass

    var superClass : ClassInfo|null
    if (d.superClass) {
      switch (d.superClass.type) {
        case 'Identifier':
          superClass = this.findClass(d.superClass.loc, d.superClass.name)
          break
        default:
          throw new TranslationError(d.superClass.loc, `Unknown superclass type ${d.superClass.type}`)
      }
    } else {
      superClass = null
    }

    const ctor = this.module.freshFunctionId(undefined)

    if (id) {
      this.classMap.set(id.name, { classConstructor: ctor})
    }

    const body : estree.ClassBody = d.body

    for (const b of body.body) {
      switch (b.type) {
        case 'MethodDefinition':
          switch (b.kind) {
            case 'constructor':
              assert(!b.static)
              assert(!b.computed)
              assert(b.key.type === 'Identifier' && b.key.name === 'constructor')
              var superMethod : mlir.Symbol|null = null
              if (superClass) {
                superMethod = superClass.classConstructor
              }
              const ts = new Scope(this, { newFunction: new FunctionContext(ctor) })
              ts.thisValue = b.static ? null : "%this"
              ts.translateFunctionExpression(b.value)
              break
            case 'method':
              {
                assert(!b.computed)
                if (b.key.type !== 'Identifier') {
                  throw new TranslationError(b.loc, `Expected method name to be identifier`)
                }
                const methodRef = this.module.freshFunctionId(b.key)
                const fun = new FunctionContext(methodRef)
                const ts = new Scope(this, { newFunction: fun })
                ts.thisValue = b.static ? null : "%this"
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
        default:
          assertNever(b)
      }
    }
  }

  translateBreakStatement(s:estree.BreakStatement) {
    if (!this.breakBlock)
      throw new TranslationError(s.loc, `'break' appears out of scope.`)
    this.endBlock(new Jump(this.breakBlock))
  }

  translateContinueStatement(s:estree.ContinueStatement) {
    if (!this.continueBlock)
      throw new TranslationError(s.loc, `'continue' appears out of scope.`)
    this.endBlock(new Jump(this.continueBlock))
  }

  // Translate return statement
  translateReturnStatement(s:estree.ReturnStatement) {
    const v = s.argument ? this.translateExpression(s.argument) : null
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
    this.endBlock(new ThrowOp(a))
  }

  // End the current block with a jump to a fresh block id
  // Return id of this block
  endBlockWithJumpToFresh():BlockId {
    const blockId = this.function.freshBlockId()
    this.startFreshBlock(new Jump(blockId), blockId)
    return blockId
  }

  private translateWhileStatement(s: estree.WhileStatement) {


    const testId = this.endBlockWithJumpToFresh()
    const v = this.translateExpression(s.test)
    const nextId = this.function.freshBlockId()

    this.startFreshBlock(new Branch(v, testId, nextId), nextId)

    let scope = new Scope(this, {breakBlock: nextId, continueBlock: testId})
    const b = scope.translateStatement(s.body)
    if (b) {
      this.endBlock(new Jump(testId))
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

(function() {
  var contents : string;
  try {
    contents = fs.readFileSync(path).toString()
  } catch (error) {
    process.stderr.write(`Could not read ${path}.\n`)
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


    const tl = new Scope(null, { })
    tl.thisValue = '%this'
//    tl.addIdent('process', {})
//    tl.addIdent('undefined', {})
//    tl.addIdent('exports', {})
//    tl.addIdent('require', {})
//    tl.addIdent('console', {})

    tl.addClass('Array',  { classConstructor: tl.module.freshFunctionId(undefined) })
    tl.addClass('Error',  { classConstructor: tl.module.freshFunctionId(undefined) })
    tl.addClass('JSON',   { classConstructor: tl.module.freshFunctionId(undefined) })
    tl.addClass('Map',    { classConstructor: tl.module.freshFunctionId(undefined) })
    tl.addClass('Object', { classConstructor: tl.module.freshFunctionId(undefined) })
    tl.addClass('Set',    { classConstructor: tl.module.freshFunctionId(undefined) })
    for (const d of s.body) {
      tl.translateStatement(d)
    }
    for (const w of tl.warnings) {
      process.stderr.write(`${formatWarning(w)}\n`)
    }
  } catch (error) {
    process.stderr.write(`Error: ${JSON.stringify(error)}\n`)
    process.stderr.write(`Error: ${error.message}\n`)

    process.exit(1)
  }
})()


