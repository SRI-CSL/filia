#include <stdio.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Parser/Parser.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/Pass.h>
#include <llvm/ADT/ImmutableMap.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallSet.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Analysis/DataFlowAnalysis.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/Dominance.h>
#include <vector>

#include "Python/PythonDialect.h"
#include "Python/PythonOps.h"

namespace {

void initContext(mlir::MLIRContext& context, const mlir::DialectRegistry& registry) {
  context.appendDialectRegistry(registry);
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::python::PythonDialect>();
}

struct Args {
  Args(int argc, const char** argv);

  // Path of file to read from
  const char* path;
};


Args::Args(int argc, const char** argv) {
  path = 0;

  for (int i = 1; i != argc; ++i) {
    if (path != 0) {
      fprintf(stderr, "Please specify only a single file to read.");
      exit(-1);
    }
    path = argv[i];
  }

  if (!path) {
    fprintf(stderr, "Please provide the path to read.\n");
    exit(-1);
  }
}

using namespace mlir;

class TaintElt {
public:
  static TaintElt join(const TaintElt& x, const TaintElt& y) {
    return TaintElt();
  }

  static TaintElt getPessimisticValueState(const Value& value) {

  }

  bool operator==(const TaintElt& o) const {
    return true;
  }

  void print(raw_ostream &os) const {

  }


};


/*
class TaintElt : public dataflow::Lattice<TaintElt> {
  enum Val { None, Untainted, Tainted, Any };

  Val value;

  TaintElt(mlir::Value val, Val v) : dataflow::AbstractSparseLattice(val), value(v) {}

public:
  /// Join the information contained in 'rhs' into this lattice. Returns
  /// if the value of the lattice changed.
  ChangeResult join(const TaintElt &rhs) override {
    switch (rhs.value) {
    case None:
      return NoChange;
    case Untainted:
      switch (value) {
      case None:
        value = Untainted;
        return Change;
      case Tainted:
        value = Any;
        return Change;
      case Untainted:
      case Any:
        return NoChange;
      }
    case Tainted:
      switch (value) {
      case None:
        value = Tainted;
        return Change;
      case Untainted:
        value = Any;
        return Change;
      case Tainted:
      case Any:
        return NoChange;
      }
    case Any:
      switch (value) {
      case Any:
        return NoChange;
      default:
        value = Any;
        return Changed;
      }
    }
  }

    /// Returns true if the lattice element is at fixpoint and further calls to
  /// `join` will not update the value of the element.
  bool isAtFixpoint() const override {
    return value == Any;
  }

  /// Mark the lattice element as having reached a pessimistic fixpoint. This
  /// means that the lattice may potentially have conflicting value states, and
  /// only the most conservative value should be relied on.
  ChangeResult markPessimisticFixpoint() override {
    value = Any;
  }

  bool operator==(const TaintElt& y) const {
    return this->value == y.value;
  }
};
  */

namespace {

using namespace mlir::dataflow;

class TaintAnalysis : public SparseDataFlowAnalysis<mlir::dataflow::Lattice<TaintElt>>  {
//public:
//  TaintAnalysis(mlir::MLIRContext* ctx) : dataflow::SparseDataFlowAnalysis<TaintElt>(ctx) {}
//  explicit SparseDataFlowAnalysis(DataFlowSolver &solver)
//      : AbstractSparseDataFlowAnalysis(solver) {}
public:
  TaintAnalysis(DataFlowSolver& s)
    : SparseDataFlowAnalysis<mlir::dataflow::Lattice<TaintElt>>(s) {
  }
/*
  /// Visit the given operation, and join any necessary analysis state
  /// into the lattices for the results and block arguments owned by this
  /// operation using the provided set of operand lattice elements (all pointer
  /// values are guaranteed to be non-null). Returns if any result or block
  /// argument value lattices changed during the visit. The lattice for a result
  /// or block argument value can be obtained by using
  /// `getLatticeElement`.
  ChangeResult
  visitOperation(Operation *op,
                 ArrayRef<LatticeElement<TaintElt>*> operands) override {
    llvm::outs() << "Visit operation: " << op->getName() << "\n";
    return ChangeResult::NoChange;
  }

  LogicalResult
  getSuccessorsForOperands(BranchOpInterface branch,
                           ArrayRef<LatticeElement<TaintElt>*> operands,
                           SmallVectorImpl<Block *> &successors) override {
    return failure();
  };
*/
//  using mlir::dataflow::SparseDataFlowAnalysis::SparseDataFlowAnalysis;

  /// Visit an operation with the lattices of its operands. This function is
  /// expected to set the lattices of the operation's results.
  void visitOperation(
        Operation *op,
        ArrayRef<const Lattice<TaintElt> *> operands,
        ArrayRef<Lattice<TaintElt> *> results) override {

  }

};

}
}


int main(int argc, const char** argv) {
  Args args(argc, argv);

  mlir::DialectRegistry registry;
  registry.insert<mlir::python::PythonDialect>();

  mlir::MLIRContext ctx;
  initContext(ctx, registry);
  mlir::Block block;
  mlir::LogicalResult r = mlir::parseSourceFile(args.path, &block, &ctx);
  if (r.failed()) {
    fprintf(stderr, "Failed to parse mlir file file.\n");
    return -1;
  }
  if (block.getOperations().size() != 1) {
    fprintf(stderr, "Missing module.\n");
    return -1;
  }

  mlir::Operation* mod = &*block.begin();

  DataFlowSolver solver;
  solver.load<TaintAnalysis>();
  if (failed(solver.initializeAndRun(mod))) {
    exit(-1);
  }

//  for (auto
//  solver.

  // Iterate through functions in module
    // Iterate through blocks in each function
      // Iterate through operations


  return 0;
}
