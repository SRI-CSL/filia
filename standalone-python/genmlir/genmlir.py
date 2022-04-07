import ast
from hashlib import new
import sys
from pprint import pprint

import mlir_python.ir as mlir
from mlir_python.dialects import (
  builtin as builtin_d,
  cf as cf_d,
  func as func_d,
  python as python_d
)

class Value:
    def __init__(self):
        return

class Analyzer(ast.NodeVisitor):

    def __init__(self, m: mlir.Module):
        self.m = m
        self.block = None
        self.map = None
        self.onDone = None

    # Internal support

    def undef_value(self):
        with mlir.InsertionPoint(self.block):
            return python_d.UndefinedOp()

    # Return the none vlaue
    def none_value(self):
        with mlir.InsertionPoint(self.block):
            return python_d.NoneOp()

    # Return the none vlaue
    def truthy(self, x: mlir.Value):
        with mlir.InsertionPoint(self.block):
            return python_d.Truthy(x)

    # Return value denoting method with given name in value
    def get_method(self, w:mlir.Value, name: str):
        with mlir.InsertionPoint(self.block):
            return python_d.GetMethod(w, mlir.StringAttr.get(name))

    # Invoke the given method.
    def invoke(self, method: mlir.Value, args: list[Value], keywords=None):
        if keywords == None or len(keywords) == 0:
            with mlir.InsertionPoint(self.block):
                return python_d.Invoke(method, args, None)
        else:
            keyAttrs = []
            for k in keywords:
                keyAttrs.append(mlir.StringAttr.get(k))
            with mlir.InsertionPoint(self.block):
                return python_d.Invoke(method, args, mlir.ArrayAttr.get(keyAttrs))

    # Invoke format value
    def format_value(self, v: Value, format: Value):
        m = self.get_method(v, '__format__')
        return self.invoke(m, [v, format])

    # Import a module and give it the given name
    def pythonImport(self, module:str, name: str):
        with mlir.InsertionPoint(self.block):
            python_d.ScopeImport(self.map, mlir.StringAttr.get(module), mlir.StringAttr.get(name))

    # Create a formatted string
    def joined_string(self, args: list[Value]):
        with mlir.InsertionPoint(self.block):
            return python_d.FormattedString(args)

    # Assign a name the value.
    def assign_name(self, name:str, v: Value):
        with mlir.InsertionPoint(self.block):
            python_d.ScopeSet(self.map, mlir.StringAttr.get(name), v)

    # Return value associated with name
    def name_value(self, name: str):
        with mlir.InsertionPoint(self.block):
            return python_d.ScopeGet(self.map, mlir.StringAttr.get(name))

    def string_constant(self, c: str):
        with mlir.InsertionPoint(self.block):
            return python_d.StrLit(mlir.StringAttr.get(c))

    def int_constant(self, c: int):
        with mlir.InsertionPoint(self.block):
            return python_d.IntLit(mlir.IntegerAttr.get(mlir.IntegerType.get_signed(64), c))

    def load_value_attribute(self, v: Value, attr: str):
        with mlir.InsertionPoint(self.block):
            get = python_d.Builtin(mlir.StringAttr.get("getattr"))
        return self.invoke(get, [v, self.string_constant(attr)])

    # Expressions
    def checked_visit_expr(self, e: ast.expr):
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

    def visit_BinOp(self, s: ast.BinOp):
        return self.undef_value() # FIXME

    def visit_Call(self, c: ast.Call):
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

    def visit_Compare(self, node: ast.Compare):
        return self.undef_value() # FIXME

    def visit_Constant(self, c: ast.Constant):
        if isinstance(c.value, str):
            return self.string_constant(c.value)
        elif isinstance(c.value, int):
            return self.int_constant(c.value)
        else:
            raise Exception(f'Unknown Constant {c.value}')

    def visit_FormattedValue(self, fv: ast.FormattedValue):
        v = self.checked_visit_expr(fv.value)
        if fv.conversion != -1:
            raise Exception(f'Conversion unsupported')
        if fv.format_spec is None:
            format = self.none_value()
        else:
            format = self.checked_visit_expr(fv.format_spec)
        return self.format_value(v, format)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> mlir.Value:
        return self.undef_value() # FIXME

    # See https://www.python.org/dev/peps/pep-0498/
    def visit_JoinedStr(self, s: ast.JoinedStr):
        args = []
        for a in s.values:
            if isinstance(a, ast.Constant):
                args.append(self.visit_Constant(a))
            elif isinstance(a, ast.FormattedValue):
                args.append(self.visit_FormattedValue(a))
            else:
                raise Exception(f'Join expression expected constant or formatted value.')
        return self.joined_string(args)

    def visit_List(self, s: ast.List):
        return self.undef_value() # FIXME

    def visit_ListComp(self, s: ast.ListComp):
        return self.undef_value() # FIXME

    def visit_Name(self, node: ast.Name):
        return self.name_value(node.id)

    def visit_Subscript(self, node: ast.Subscript):
        return self.undef_value() #FIXME

    def visit_Tuple(self, node: ast.Subscript):
        return self.undef_value() #FIXME

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

    def assignLhs(self, tgt, v):
        if isinstance(tgt, ast.Tuple):
            n = len(tgt.elts)
            with mlir.InsertionPoint(self.block):
                python_d.TupleCheck(v, mlir.IntegerAttr.get(mlir.IntegerType.get_signed(64), n))
            for i in range(0, n):
                with mlir.InsertionPoint(self.block):
                    sv = python_d.TupleGet(v, mlir.IntegerAttr.get(mlir.IntegerType.get_signed(64), i))
                self.assignLhs(tgt.elts[i], sv)
        elif isinstance(tgt, ast.Name):
            self.assign_name(tgt.id, v)
        else:
            raise Exception(f'Unexpected target {tgt.__class__.__name__}')

    def visit_Assign(self, a: ast.Assign):
        if len(a.targets) != 1:
            raise Exception('Assignment must have single left-hand side.')
        tgt = a.targets[0]
        r = self.checked_visit_expr(a.value)
        self.assignLhs(tgt, r)
        return True

    def visit_AugAssign(self, node: ast.AugAssign):
        #FIXME
        return True

    def visit_Expr(self, e: ast.Expr):
        r = self.checked_visit_expr(e.value)
        return True

    def visit_For(self, s: ast.For) -> bool:
        r = self.checked_visit_expr(s.target)

        if len(s.orelse) > 0:
            raise Exception('For loop with else clause not yet supported.')

        return True # FIXME

    def visit_FunctionDef(self, s: ast.FunctionDef):
        scopeType = python_d.ScopeType.get()
        valueType = python_d.ValueType.get()
        with mlir.InsertionPoint(self.m.body):
            tp = mlir.FunctionType.get([scopeType], [valueType])
            fun = builtin_d.FuncOp(s.name, tp) # FIXME: Use a name generator to avoid collisions

        funAnalyzer = Analyzer(self.m)
        funAnalyzer.block = mlir.Block.create_at_start(fun.regions[0], [scopeType])

        with mlir.InsertionPoint(funAnalyzer.block):
            funAnalyzer.map = python_d.ScopeExtend(funAnalyzer.block.arguments[0])

        cont = funAnalyzer.visitStmts(s.body)
        if cont:
            with mlir.InsertionPoint(funAnalyzer.block):
                func_d.ReturnOp([funAnalyzer.none_value()])

        return True

    def visit_If(self, s: ast.If):
        c = self.truthy(self.checked_visit_expr(s.test))

        newBlock = self.block.create_after()

        if s.orelse:
            falseBlock = self.block.create_after()
            falseAnalyzer = Analyzer(self.m)
            falseAnalyzer.block = falseBlock
            falseAnalyzer.map = self.map
            if falseAnalyzer.visitStmts(s.orelse):
                with mlir.InsertionPoint(falseAnalyzer.block):
                    cf_d.BranchOp([], newBlock)
        else:
            falseBlock = newBlock

        if s.body:
            trueBlock = self.block.create_after()
            trueAnalyzer = Analyzer(self.m)
            trueAnalyzer.block = trueBlock
            trueAnalyzer.map = self.map
            if trueAnalyzer.visitStmts(s.body):
                with mlir.InsertionPoint(trueAnalyzer.block):
                    cf_d.BranchOp([], newBlock)
        else:
            trueBlock = newBlock

        with mlir.InsertionPoint(self.block):
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
            rets = [self.checked_visit_expr(node.value)]
        else:
            rets = []

        with mlir.InsertionPoint(self.block):
            func_d.ReturnOp(rets)
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
                self.assign_name(var.id, r)
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
        with mlir.InsertionPoint(self.m.body):
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

    def visit_While(self, node: ast.While) -> bool:
        return True # FIXME

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
        analyzer = Analyzer(m)
        r = analyzer.visit(tree)
    assert (r is not None)
    print(str(m))

if __name__ == "__main__":
    main()