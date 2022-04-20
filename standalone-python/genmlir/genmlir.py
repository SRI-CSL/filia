import ast
import sys

import mlir_python.ir as mlir
from mlir_python.dialects import (
  builtin as builtin_d,
  cf as cf_d,
  func as func_d,
  python as python_d
)

# Return value associated with name
def scope_get(block: mlir.Block, map:mlir.Value, name: str) -> mlir.Value:
    with mlir.InsertionPoint(block):
        return python_d.ScopeGet(map, mlir.StringAttr.get(name))

# Assign a name the value.
def scope_set(block: mlir.Block, map:mlir.Value, name:str, v: mlir.Value) -> mlir.Value:
    with mlir.InsertionPoint(block):
        python_d.ScopeSet(map, mlir.StringAttr.get(name), v)

def scope_set_lhs(block: mlir.Block, map:mlir.Value, tgt: ast.expr, v:mlir.Value):
    if isinstance(tgt, ast.Tuple):
        n = len(tgt.elts)
        with mlir.InsertionPoint(block):
            python_d.TupleCheck(v, mlir.IntegerAttr.get(mlir.IntegerType.get_signed(64), n))
        for i in range(0, n):
            with mlir.InsertionPoint(block):
                sv = python_d.TupleGet(v, mlir.IntegerAttr.get(mlir.IntegerType.get_signed(64), i))
            scope_set_lhs(block, map, tgt.elts[i], sv)
    elif isinstance(tgt, ast.Name):
        scope_set(block, map, tgt.id, v)
    else:
        raise Exception(f'Unexpected target {tgt.__class__.__name__} {tgt.lineno}:{tgt.col_offset}')

# invoke_next(next, nextBlock, doneBlock, throwBlock) invokes the next method and returns
# a pair (bodyBlock, value) where:
# * bodyBlock is a fresh block and
# * value is a value within bodyBlock that refers to the next value in the iterator.
#
# * next is the method called to invoke next.
# * nextBlock is the block that new blocks are appended too
# *
def invoke_next(next: mlir.Value, curBlock: mlir.Block, doneBlock: mlir.Block, throwBlock: mlir.Block):
    valueType = python_d.ValueType.get()

    nextBlock = curBlock.create_after()
    with mlir.InsertionPoint(curBlock):
        cf_d.BranchOp([], nextBlock)

    bodyBlock   = nextBlock.create_after(valueType)
    exceptBlock = nextBlock.create_after(valueType)

    with mlir.InsertionPoint(nextBlock):
        python_d.Invoke(next, [], None, [], [], bodyBlock, exceptBlock)

    exception   = exceptBlock.arguments[0]
    with mlir.InsertionPoint(exceptBlock):
        c = python_d.IsInstance(exception, mlir.StringAttr.get("StopIteration"))
        cf_d.CondBranchOp(c, [], [exception], doneBlock, throwBlock)

    return nextBlock, bodyBlock, bodyBlock.arguments[0]

# Return the none vlaue
def truthy(block: mlir.Block, x: mlir.Value) -> mlir.Value:
    with mlir.InsertionPoint(block):
        return python_d.Truthy(x)


class Module:

    def __init__(self, mlir:mlir.Module):
        self.mlir = mlir
        self.vars = {}

    # Ensure a symbol has a fresh name.
    def fresh_symbol(self, nm: str|None) -> str:
        if nm == None:
            nm = "_mlir_gen"
        else:
            # Strip out @ sign as it is used for marking count.
            nm = nm.replace('@', '')
            if nm == "":
                nm = "_mlir_gen"
        cnt = self.vars.get(nm)
        if cnt == None:
            self.vars[nm] = 0
            return nm
        else:
            self.vars[nm] = cnt + 1
            return f'{nm}@{cnt}'

class Analyzer(ast.NodeVisitor):

    def __init__(self, m: Module):
        self.module = m
        self.block = None
        self.map = None
        self.onDone = None
        self.exceptBlock = None

    # Internal support

    def undef_value(self, e: ast.AST) -> mlir.Value:
        sys.stderr.write(f'Unsupported value {e.__class__.__name__} at {e.lineno}:{e.col_offset}\n')
        with mlir.InsertionPoint(self.block):
            return python_d.UndefinedOp()

    # Return the none vlaue
    def none_value(self) -> mlir.Value:
        with mlir.InsertionPoint(self.block):
            return python_d.NoneOp()

    # Return value denoting method with given name in value
    def get_method(self, w:mlir.Value, name: str) -> mlir.Value:
        with mlir.InsertionPoint(self.block):
            return python_d.GetMethod(w, mlir.StringAttr.get(name))


    def get_except_block(self):
        if self.exceptBlock != None:
           return self.exceptBlock

        valueType = python_d.ValueType.get()
        exceptBlock = self.block.create_after(valueType)
        with mlir.InsertionPoint(exceptBlock):
            python_d.Throw(exceptBlock.arguments[0])

        self.exceptBlock = exceptBlock
        return exceptBlock

    # Invoke the given method.
    def invoke(self, method: mlir.Value, args: list[mlir.Value], keywords=None) -> mlir.Value:
        if keywords == None or len(keywords) == 0:
            keyAttr = None
        else:
            keyAttrs = []
            for k in keywords:
                keyAttrs.append(mlir.StringAttr.get(k))
            keyAttr = mlir.ArrayAttr.get(keyAttrs)

        valueType = python_d.ValueType.get()
        exceptBlock = self.get_except_block()
        returnBlock = self.block.create_after(valueType)

        with mlir.InsertionPoint(self.block):
            python_d.Invoke(method, args, keyAttr, [], [], returnBlock, exceptBlock)

        self.block = returnBlock
        return returnBlock.arguments[0]

    # Invoke format value
    def format_value(self, v: mlir.Value, format: mlir.Value) -> mlir.Value:
        m = self.get_method(v, '__format__')
        return self.invoke(m, [v, format])

    # Import a module and give it the given name
    def pythonImport(self, module:str, name: str) -> mlir.Value:
        with mlir.InsertionPoint(self.block):
            python_d.ScopeImport(self.map, mlir.StringAttr.get(module), mlir.StringAttr.get(name))

    # Create a formatted string
    def joined_string(self, args: list[mlir.Value]) -> mlir.Value:
        with mlir.InsertionPoint(self.block):
            return python_d.FormattedString(args)


    def string_constant(self, c: str) -> mlir.Value:
        with mlir.InsertionPoint(self.block):
            return python_d.StrLit(mlir.StringAttr.get(c))

    def int_constant(self, c: int) -> mlir.Value:
        with mlir.InsertionPoint(self.block):
            if -2**63 <= c and c < 2**63:
                return python_d.S64Lit(mlir.IntegerAttr.get(mlir.IntegerType.get_signed(64), c))
            else:
                return python_d.IntLit(mlir.StringAttr.get(str(c)))

    def builtin(self, name: str):
        with mlir.InsertionPoint(self.block):
            return python_d.Builtin(mlir.StringAttr.get(name))

    def load_value_attribute(self, v: mlir.Value, attr: str):
        get = self.builtin("getattr")
        return self.invoke(get, [v, self.string_constant(attr)])

    def returnOp(self, v: mlir.Value):
        with mlir.InsertionPoint(self.block):
            func_d.ReturnOp([v])

    def create_fun(self, name: str, args: ast.arguments):
        scopeType = python_d.ScopeType.get()
        valueType = python_d.ValueType.get()

        assert(args.vararg == None)
        assert(len(args.kwonlyargs) == 0)
        assert(len(args.kw_defaults) == 0)
        assert(args.kwarg == None)
        assert(len(args.defaults) == 0)
        arg_count = len(args.args)

        symbol_name = self.module.fresh_symbol(name)

        with mlir.InsertionPoint(self.module.mlir.body):
            argTypes = [scopeType] + arg_count * [valueType]
            tp = mlir.FunctionType.get(argTypes, [valueType])
            fun = builtin_d.FuncOp(symbol_name, tp)

        fun_analyzer = Analyzer(self.module)
        fun_analyzer.block = mlir.Block.create_at_start(fun.regions[0], argTypes)

        with mlir.InsertionPoint(fun_analyzer.block):
            fun_analyzer.map = python_d.ScopeExtend(fun_analyzer.block.arguments[0])

        # Initialize scope from function arguments.
        arg_names = []
        for i in range(arg_count):
            name = args.args[i].arg
            value = fun_analyzer.block.arguments[1+i]
            arg_names.append(mlir.StringAttr.get(name))
            scope_set(fun_analyzer.block, fun_analyzer.map, name, value)

        symbol_attr = mlir.FlatSymbolRefAttr.get(symbol_name)
        arg_attrs = mlir.ArrayAttr.get(arg_names)
        with mlir.InsertionPoint(self.block):
            fun_value = python_d.FunctionRef(symbol_attr, arg_attrs, self.map)

        return fun_analyzer, fun_value

    # Expressions
    def checked_visit_expr(self, e: ast.expr) -> mlir.Value:
        r = self.visit(e)
        if r == None:
            raise Exception(f'Unsupported expression {type(e)}')
        return r

    def visit_Attribute(self, a: ast.Attribute):
        val = self.checked_visit_expr(a.value)
        if isinstance(a.ctx, ast.Load):
            return self.load_value_attribute(val, a.attr)
        elif isinstance(a.ctx, ast.Store):
            raise Exception(f'Store attribute unsupported.')
        elif isinstance(a.ctx, ast.Del):
            raise Exception(f'Delete attribute unsupported.')
        else:
            raise Exception(f'Unknown context {type(a.ctx)}')

    # FIXME Make static
    operator_map = {
        ast.Add: python_d.AddOp,
        ast.BitAnd: python_d.BitAndOp,
        ast.BitOr: python_d.BitOrOp,
        ast.BitXor: python_d.BitXorOp,
        ast.Div: python_d.DivOp,
        ast.FloorDiv: python_d.FloorDivOp,
        ast.LShift: python_d.LShiftOp,
        ast.Mod: python_d.ModOp,
        ast.Mult: python_d.MultOp,
        ast.MatMult: python_d.MatMultOp,
        ast.Pow: python_d.PowOp,
        ast.RShift: python_d.RShiftOp,
        ast.Sub: python_d.SubOp
    }

    def apply_binop(self, left: mlir.Value, op, right: mlir.Value) -> mlir.Value:
        valueType = python_d.ValueType.get()
        exceptBlock = self.get_except_block()
        returnBlock = self.block.create_after(valueType)

        with mlir.InsertionPoint(self.block):
            op(left, right, [], [], returnBlock, exceptBlock)

        self.block = returnBlock
        return returnBlock.arguments[0]

    def visit_BinOp(self, e: ast.BinOp) -> mlir.Value:
        op = self.operator_map.get(e.op.__class__)
        if op == None:
            return self.undef_value(e)

        left  = self.checked_visit_expr(e.left)
        right = self.checked_visit_expr(e.right)
        return self.apply_binop(left, op, right)

    def visit_Call(self, c: ast.Call) -> mlir.Value:
        f = self.visit(c.func)
        assert(f != None)
        args = []
        for a in c.args:
            args.append(self.checked_visit_expr(a))

        keywords = []
        for k in c.keywords:
            if k.arg == None:
                raise Exception(f'Did not expect ** in call')
            args.append(self.checked_visit_expr(a))
            keywords.append(k.arg)
        return self.invoke(f, args, keywords)


    # FIXME Make static
    comparison_map = {
        ast.Eq : python_d.EqOp,
        ast.Gt : python_d.GtOp,
        ast.GtE : python_d.GtEOp,
        ast.In : python_d.InOp,
        ast.Is : python_d.IsOp,
        ast.IsNot : python_d.IsNotOp,
        ast.Lt : python_d.LtOp,
        ast.LtE : python_d.LtEOp,
        ast.NotEq : python_d.NotEqOp,
        ast.NotIn : python_d.NotInOp
    }

    def visit_Compare(self, e: ast.Compare) -> mlir.Value:
        left  = self.checked_visit_expr(e.left)
        args = e.comparators.__iter__()
        for cmpop in e.ops:
            right = self.checked_visit_expr(args.__next__())
            op = self.comparison_map.get(cmpop.__class__)
            if op == None:
                left = self.undef_value(e)
            else:
                left = self.apply_binop(left, op, right)
        return left

    def visit_Constant(self, c: ast.Constant) -> mlir.Value:
        if isinstance(c.value, str):
            return self.string_constant(c.value)
        elif isinstance(c.value, int):
            r = self.int_constant(c.value)
            if r == None:
                raise Exception(f'Integer constant {c.value} at {c.lineno}:{c.col_offset} is out of range.')
            return r
        else:
            raise Exception(f'Unknown Constant {c.value}')

    def visit_FormattedValue(self, fv: ast.FormattedValue) -> mlir.Value:
        v = self.checked_visit_expr(fv.value)
        if fv.conversion != -1:
            raise Exception(f'Conversion unsupported')
        if fv.format_spec is None:
            format = self.none_value()
        else:
            format = self.checked_visit_expr(fv.format_spec)
        return self.format_value(v, format)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> mlir.Value:
        return self.undef_value(node) # FIXME

    # See https://www.python.org/dev/peps/pep-0498/
    def visit_JoinedStr(self, s: ast.JoinedStr) -> mlir.Value:
        args = []
        for a in s.values:
            if isinstance(a, ast.Constant):
                args.append(self.visit_Constant(a))
            elif isinstance(a, ast.FormattedValue):
                args.append(self.visit_FormattedValue(a))
            else:
                raise Exception(f'Join expression expected constant or formatted value.')
        return self.joined_string(args)

    def visit_Lambda(self, e: ast.Lambda) -> mlir.Value:
        funAnalyzer, fun_value = self.create_fun("_lambda", e.args)
        funAnalyzer.returnOp(funAnalyzer.checked_visit_expr(e.body))
        return fun_value

    def visit_List(self, e: ast.List) -> mlir.Value:
        args = map(self.checked_visit_expr, e.elts)
        with mlir.InsertionPoint(self.block):
            return python_d.List(args)

    def visit_ListComp(self, e: ast.ListComp) -> mlir.Value:
        # Get identifier of block to throw
        throwBlock = self.get_except_block()

        #
        orig_map = self.map
        with mlir.InsertionPoint(self.block):
            # Variables in comprehension do not escape.
            self.map = python_d.ScopeExtend(orig_map)
            # Create empty list for storing result.
            r = python_d.List([])

        # Get append method off of list.
        append = self.get_method(r, 'append')

        # Create block for evaluating expression
        finalBlock = self.block.create_after()
        doneBlock = finalBlock
        for g in e.generators:
            assert(not g.is_async)
            assert(len(g.ifs) == 0)

            l = self.checked_visit_expr(g.iter)
            i = self.invoke(self.get_method(l, '__iter__'), [])
            next = self.get_method(i, '__next__')

            nextBlock, self.block, bodyValue = invoke_next(next, self.block, doneBlock, throwBlock)
            scope_set_lhs(self.block, self.map, g.target, bodyValue)
            # Map done block to next so other loop.
            doneBlock = nextBlock

        e = self.checked_visit_expr(e.elt)
        self.invoke(append, [e])

        with mlir.InsertionPoint(self.block):
            cf_d.BranchOp([], doneBlock)

        self.block = finalBlock
        self.map = orig_map

        return l

    def visit_Name(self, node: ast.Name):
        return scope_get(self.block, self.map, node.id)

    def visit_Slice(self, e: ast.Slice):
        upper = self.checked_visit_expr(e.upper) if e.upper != None else self.none_value()

        sliceFn = self.builtin('slice')
        if e.step != None:
            lower = self.checked_visit_expr(e.lower) if e.lower != None else self.none_value()
            step = self.checked_visit_expr(e.step)
            return self.invoke(sliceFn, [lower, upper, step])
        elif e.lower != None:
            lower = self.checked_visit_expr(e.lower)
            return self.invoke(sliceFn, [lower, upper])
        else:
            return self.invoke(sliceFn, [upper])

    def visit_Subscript(self, e: ast.Subscript):
        v = self.checked_visit_expr(e.value)
        slice = self.checked_visit_expr(e.slice)
        getitem = self.get_method(v, '__getitem__')
        return self.invoke(getitem, [slice])

    def visit_Tuple(self, e: ast.Tuple):
        args = map(self.checked_visit_expr, e.elts)
        with mlir.InsertionPoint(self.block):
            return python_d.Tuple(args)

    def visit_UnaryOp(self, e: ast.UnaryOp) -> mlir.Value:
        return self.undef_value(e) # FIXME

    # Statements
    def checked_visit_stmt(self, e: ast.stmt):
        r = self.visit(e)
        if r == None:
            raise Exception(f'Unsupported expression {type(e)}')
        return r

    def visitStmts(self, stmts):
        for stmt in stmts:
            if not self.checked_visit_stmt(stmt):
                return False
        return True

    def visit_Assign(self, a: ast.Assign):
        if len(a.targets) != 1:
            raise Exception('Assignment must have single left-hand side.')
        tgt = a.targets[0]
        r = self.checked_visit_expr(a.value)
        scope_set_lhs(self.block, self.map, tgt, r)
        return True

    def visit_AugAssign(self, node: ast.AugAssign):
        #FIXME
        return True

    def visit_Expr(self, e: ast.Expr):
        r = self.checked_visit_expr(e.value)
        return True

    def visit_For(self, s: ast.For) -> bool:
        if len(s.orelse) > 0:
            raise Exception('For orelse unsupported.')
        r = self.checked_visit_expr(s.iter)
        enter = self.get_method(r, '__iter__')
        i = self.invoke(enter, [])
        next = self.get_method(i, '__next__')


        throwBlock = self.get_except_block()

        doneBlock = self.block.create_after()

        nextBlock, bodyBlock, value = invoke_next(next, self.block, doneBlock, throwBlock)
        scope_set_lhs(bodyBlock, self.map, s.target, value)

        # Get value for loop
        self.block = bodyBlock
        bodyCont = self.visitStmts(s.body)
        if bodyCont:
            with mlir.InsertionPoint(self.block):
                cf_d.BranchOp([], nextBlock)

        self.block = doneBlock
        return True

    def visit_FunctionDef(self, s: ast.FunctionDef):
        funAnalyzer, fun_value = self.create_fun(s.name, s.args)

        cont = funAnalyzer.visitStmts(s.body)
        if cont:
            funAnalyzer.returnOp(funAnalyzer.none_value())

        scope_set(self.block, self.map, s.name, fun_value)

        return True

    def visit_If(self, s: ast.If):
        test = self.checked_visit_expr(s.test)
        c = truthy(self.block, test)

        initBlock = self.block
        newBlock = self.block.create_after()


        if s.orelse:
            falseBlock = initBlock.create_after()
            self.block = falseBlock
            if self.visitStmts(s.orelse):
                with mlir.InsertionPoint(self.block):
                    cf_d.BranchOp([], newBlock)
        else:
            falseBlock = newBlock

        if s.body:
            trueBlock = initBlock.create_after()
            self.block = trueBlock
            if self.visitStmts(s.body):
                with mlir.InsertionPoint(self.block):
                    cf_d.BranchOp([], newBlock)
        else:
            trueBlock = newBlock

        with mlir.InsertionPoint(initBlock):
            cf_d.CondBranchOp(c, [], [], trueBlock, falseBlock)
        self.block = newBlock
        return True

    def visit_Import(self, s: ast.Import):
        for a in s.names:
            self.pythonImport(a.name, a.asname or a.name)
        return True

    def visit_Return(self, node: ast.Return):
        if self.onDone:
            self.onDone()

        if node.value != None:
            ret = self.checked_visit_expr(node.value)
        else:
            ret = self.funAnalyzer.none_value()
        self.returnOp(ret)

        return False

    # See https://docs.python.org/3/reference/compound_stmts.html#with
    def visit_With(self, w: ast.With):
        #FIXME. Add try/finally blocks
        exitMethods = []
        for item in w.items:
            ctx = self.checked_visit_expr(item.context_expr)
            assert(ctx != None)
            enter = self.get_method(ctx, '__enter__')
            exit = self.get_method(ctx, '__exit__')
            r  = self.invoke(enter, [])
            var = item.optional_vars
            if var != None:
                assert(isinstance(var, ast.Name))
                scope_set(self.block, self.map, var.id, r)
            exitMethods.append(exit)
        prevDone = self.onDone
        def onDone():
            for exit in reversed(exitMethods):
                self.invoke(exit, [])
            if prevDone:
                prevDone()
        self.onDone = onDone
        cont = self.visitStmts(w.body)
        self.onDone = prevDone
        if cont:
            onDone()
        return cont

    def visit_Module(self, m: ast.Module):
        with mlir.InsertionPoint(self.module.mlir.body):
            tp = mlir.FunctionType.get([], [])
            script_main = builtin_d.FuncOp("script_main", tp)

        self.block = mlir.Block.create_at_start(script_main.regions[0])

        with mlir.InsertionPoint(self.block):
            self.map = python_d.ScopeInit()

        cont = self.visitStmts(m.body)
        if cont:
            with mlir.InsertionPoint(self.block):
                func_d.ReturnOp([])
        return True

    def visit_While(self, s: ast.While) -> bool:
        if len(s.orelse) > 0:
            raise Exception('While orelse unsupported.')

        testBlock = self.block.create_after()
        with mlir.InsertionPoint(self.block):
            cf_d.BranchOp([], testBlock)

        self.block = testBlock
        testExpr = self.checked_visit_expr(s.test)
        c = truthy(self.block, testExpr)
        bodyBlock = self.block.create_after()
        doneBlock = bodyBlock.create_after()
        with mlir.InsertionPoint(self.block):
            cf_d.CondBranchOp(c, [], [], bodyBlock, doneBlock)

        # Execute loop body
        self.block = bodyBlock
        bodyCont = self.visitStmts(s.body)
        if bodyCont:
            with mlir.InsertionPoint(self.block):
                cf_d.BranchOp([], testBlock)

        self.block = doneBlock
        return True

def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Please specify input file.\n")
        sys.exit(-1)
    path = sys.argv[1]
    with open(path, "r") as source:
        tree = ast.parse(source.read())

    with mlir.Context() as ctx, mlir.Location.file("f.mlir", line=42, col=1, context=ctx):
        python_d.register_dialect()
#        ctx.allow_unregistered_dialects = True
        m = mlir.Module.create()
        analyzer = Analyzer(Module(m))
        r = analyzer.visit(tree)
    assert (r is not None)
    print(str(m))

if __name__ == "__main__":
    main()