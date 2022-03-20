import ast
from pprint import pprint
import mlir.ir as mlir
import mlir.dialects.builtin as mlirb
import mlir.dialects.cf as mlir_cf
import mlir.dialects.func as mlir_func

mlirb.FuncOp
# Context, Location, Module

class Value:
    def __init__(self):
        return

class Analyzer(ast.NodeVisitor):

    def __init__(self, m: mlir.Module):
        self.m = m

    # Internal support

    # Return value denoting method with given name in value
    def get_method(self, w: Value, name: str):
        return Value() #FIXME

    # Invoke the given method.
    def invoke(self, method: Value, args: list[Value]):
        return Value() #FIXME

    def load_value_attribute(self, v: Value, attr: str):
        return Value() # FIXME

    def joined_string(self, args: list[Value]):
        return Value() # FIXME

    # Assign a name the value.
    def assign_name(self, name, v: Value):
        return None #FIXME

    # Return value associated with name
    def name_value(self, name: str):
        return Value() #FIXME

    def string_constant(self, c: str):
        # FIXME
        return Value()

    def int_constant(self, c: int):
        # FIXME
        return Value()

    def format_value(self, v: Value):
        #FIXME
        return Value()

    # Expressions
    def checked_visit_expr(self, e: ast.expr):
        r = self.visit(e)
        if not isinstance(r, Value):
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

    def visit_Call(self, c: ast.Call):
        f = self.visit(c.func)
        assert(f != None)
        args = []
        assert len(c.keywords) == 0
        for a in c.args:
            args.append(self.checked_visit_expr(a))
        return self.invoke(f, args)

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
        if fv.format_spec is not None:
            raise Exception(f'Format spec is expr')
        # FIXME
        return self.format_value(v)

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

    def visit_Name(self, node: ast.Name):
        return self.name_value(node.id)

    # Statements
    def checked_visit_stmt(self, e: ast.stmt):
        r = self.visit(e)
        if r != True:
            raise Exception(f'Unsupported expression {type(e)}')
        return

    def visit_Assign(self, a: ast.Assign):
        if len(a.targets) != 1:
            raise Exception('Assignment must have single left-hand side.')
        assert(isinstance(a.targets[0], ast.Name))
        name = a.targets[0]
        r = self.checked_visit_expr(a.value)
        self.assign_name(name.id, r)
        return True

    def visit_Expr(self, e: ast.Expr):
        r = self.checked_visit_expr(e.value)
        return True

    # See https://docs.python.org/3/reference/compound_stmts.html#with
    def visit_With(self, w: ast.With):
        #FIXME. Add try/finally blocks
        exitMethods = []
        for item in w.items:
            ctx = self.visit(item.context_expr)
            assert(ctx != None)
            enter = self.get_method(ctx, '__enter__')
            exit = self.get_method(ctx, '__exit__')
            r  = self.invoke(enter, [])
            var = item.optional_vars
            if var != None:
                assert(isinstance(var, ast.Name))
                self.assign_name(var.id, r)
            exitMethods.append(exit)
        for stmt in w.body:
            self.checked_visit_stmt(stmt)
        for exit in reversed(exitMethods):
            self.invoke(exit, [])
        return True

    def visit_Module(self, m: ast.Module):
        with mlir.InsertionPoint(self.m.body):
            tp = mlir.FunctionType.get([], [])
            script_main = mlirb.FuncOp("script_main", tp)

        self.block = mlir.Block.create_at_start(script_main.regions[0])

        for stmt in m.body:
            self.checked_visit_stmt(stmt)

        with mlir.InsertionPoint(self.block):
            mlir_func.ReturnOp([])
        return True

def main():

    with open("example/insecure_eval.py", "r") as source:
        tree = ast.parse(source.read())

    with mlir.Context() as ctx, mlir.Location.file("f.mlir", line=42, col=1, context=ctx):
        m = mlir.Module.create()

        analyzer = Analyzer(m)
        r = analyzer.visit(tree)
        assert (r is not None)

        print("Module")
        x = m.dump()
        print(x)


if __name__ == "__main__":
    main()