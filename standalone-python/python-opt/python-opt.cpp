#include <stdio.h>
#include <mlir/Parser.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/Pass.h>
#include <llvm/ADT/ImmutableMap.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallSet.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <vector>

#include "Python/PythonDialect.h"
#include "Python/PythonOps.h"

#include "PythonDomain.h"
#include "ValueTranslator.h"

//using ValueArgumentMap = llvm::DenseMap<mlir::Value, unsigned>;

using ArgMap = llvm::DenseMap<ScopeField, unsigned>;

using BlockValueMap = llvm::DenseMap<mlir::Block*, llvm::DenseSet<mlir::Value>>;

/**
 * Walk through all operations in a list of blocks to populate map from blocks to defined
 * values.
 *
 * @param definedValues Map to populate
 * @param blks Blocks in a function
 */
static
void populateDefinedValues(BlockValueMap& definedValues, mlir::Region* body) {
  // Populate defined values
  auto& blks = body->getBlocks();
  for (auto blk = blks.begin(); blk != blks.end(); ++blk) {
    auto p = definedValues.insert(std::make_pair(&*blk, llvm::DenseSet<mlir::Value>()));
    auto& s = p.first->second;

    for (auto op = blk->begin(); op != blk->end(); ++op) {
      const auto& res = op->getResults();
      for (const auto& r : res) {
        s.insert(r);
      }
    }
  }
}

/**
 * Walk through all operations in a list of blocks to populate map from blocks to values
 * inherited from immediate dominators.
 */
static
void populateInheritedValues(mlir::DominanceInfo& domInfo, BlockValueMap& inheritedValues, const BlockValueMap& definedValues, mlir::Region* body) {
  // Initialize inheritedValues
  if (body->hasOneBlock()) return;

  auto& domTree = domInfo.getDomTree(body);
  auto& blks = body->getBlocks();
  // Populate inherited values given defined values
  for (auto blk = blks.begin(); blk != blks.end(); ++blk) {
    auto p = inheritedValues.insert(std::make_pair(&*blk, llvm::DenseSet<mlir::Value>()));
    auto& s = p.first->second;
    auto node = domTree.getNode(&*blk);
    if (!node) continue;

    node = node->getIDom();
    while (node) {
      auto domBlock = node->getBlock();
      auto domDefValues = definedValues.find(domBlock);
      if (domDefValues != definedValues.end()) {
        auto defSet = domDefValues->second;
        for (auto v : defSet) {
          s.insert(v);
        }
      }
      node = node->getIDom();
    }
  }
}

/**
 * This maps how values in one block are translated into a successor.
 */
class FunctionValueTransitionMap {
public:
  // Maps MLIR values to index of argument they appear in block.
  using SuccessorMap = std::vector<std::pair<mlir::Block*, mlir::MutableOperandRange>>;
private:

  // Maps each block to the values defined in the block.
  BlockValueMap definedValues;

  // Maps each block to the set of values defined when the block stars.
  BlockValueMap inheritedValues;

  // Build map from blocks to a map that maps successors to the value for block
  // arguments.
  llvm::DenseMap<mlir::Block*, SuccessorMap> transMap;

  // Maps blocks to the numb
  llvm::DenseMap<mlir::Block*, unsigned> argCount;

  /**
   * Adds a jump to the target block with the arguments to successor map.
   * N.B.  The offset indicates the number of argumments the successor block
   * provides with the remaining arguments provided by the operands.
   *
   * @param m Map to update
   * @param offset Offset of first argument in successor blocks args is for.
   * @param target Target block
   * @param operands
   */
  static
  void addSuccessorEdge(SuccessorMap& m, unsigned offset, mlir::Block* target, mlir::MutableOperandRange operands) {
    m.push_back(std::make_pair(target, operands));
  }

  template<typename T>
  static bool add_call_edges(SuccessorMap& m, mlir::Operation* opPtr) {
    if (!mlir::isa<T>(opPtr))
      return false;

    auto op = mlir::cast<T>(opPtr);
    addSuccessorEdge(m, 1, op.returnDest(), op.returnDestOperandsMutable());
    addSuccessorEdge(m, 1, op.exceptDest(), op.exceptDestOperandsMutable());
    return true;
  }

public:
  FunctionValueTransitionMap(mlir::FuncOp fun, mlir::DominanceInfo& domInfo) {
    ::mlir::Region* body = &fun.body();
    populateDefinedValues(definedValues, body);
    populateInheritedValues(domInfo, inheritedValues, definedValues, body);

    auto& blks = body->getBlocks();
    // Populate inherited values given defined values
    for (auto blk = blks.begin(); blk != blks.end(); ++blk) {
      auto& s = transMap.insert(std::make_pair(&*blk, SuccessorMap())).first->second;

      auto opPtr = &blk->back();
      if (add_call_edges<mlir::python::Invoke>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::InvertOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::NotOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::UAddOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::USubOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::AddOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::BitAndOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::BitOrOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::BitXorOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::DivOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::FloorDivOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::LShiftOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::ModOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::MultOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::MatMultOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::PowOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::RShiftOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::SubOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::EqOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::GtOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::GtEOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::InOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::IsOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::IsNotOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::LtOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::LtEOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::NotEqOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::NotInOp>(s, opPtr)) {
      } else if (mlir::isa<mlir::python::ThrowOp>(opPtr)) {
      } else if (mlir::isa<mlir::func::ReturnOp>(opPtr)) {
        // Do nothing on return
      } else {
        fprintf(stderr, "Unsupported terminal operator: ");
        opPtr->getName().print(::llvm::errs());
        fprintf(stderr, "\n");
        exit(-1);
      }
    }
  }

  const InheritedSet& getInheritedValues(mlir::Block* block) {
    auto i = inheritedValues.find(block);
    if (i == inheritedValues.end()) {
      fatal_error("Could not find block.");
    }
    return i->second;
  }

  const SuccessorMap& succMap(mlir::Block* block) {
    auto i = transMap.find(block);
    if (i == transMap.end()) {
      fatal_error("Could not find block.");
    }
    return i->second;
  }
};

class ValueTranslator;

namespace {

void initContext(mlir::MLIRContext& context, const mlir::DialectRegistry& registry) {
  context.appendDialectRegistry(registry);
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::python::PythonDialect>();
}

/**
 * This contains information needed to calculate block domains
 *
 */
class BlockInvariantFixpointQueue {
public:
  // Maps blocks to the extra arguments for the block.
  // The entry for the block is true if the argument is still
  // used and false otherwise.
  llvm::DenseMap<mlir::Block*, BlockArgInfo> blockMap;

private:
  std::vector<BlockArgInfo*> pending;

  // Add a new block and locals domain.
  BlockArgInfo* addNew(mlir::Block* b) {
    auto p = blockMap.insert(std::make_pair(b, BlockArgInfo(b)));
    auto r = &p.first->second;
    pending.push_back(r);
    return r;
  }

public:
  BlockInvariantFixpointQueue(mlir::Block* entry) {
    addNew(entry);
  }

  /**
   * Return true if there is another block to process.
   */
  bool hasNext() {
    return !pending.empty();
  }

  /**
   * Get next block and initial abstract state.
   */
  BlockArgInfo* nextBlock() {
    auto p = pending.back();
    pending.pop_back();
    return p;
  }

  /**
   * Update local domains of all the successors for a given block.
   *
   * @param m Information about values defined in each block.
   * @param src Block to update successors of
   * @param term Abstract domain state for block at end.
   */
  void updateSuccessors(FunctionValueTransitionMap& m,
                        mlir::Block* src,
                        const LocalsDomain& term);

  /**
   * Update local domains of all the successors for a given block.
   *
   * @param m Information about values defined in each block.
   * @param src Block to update successors of
   * @param term Abstract domain state for block at end.
   */
  void addSuccessors(FunctionValueTransitionMap& m,
                     mlir::Block* src,
                     mlir::OpBuilder& builder,
                     const std::vector<mlir::Value> argValues,
                     const LocalsDomain& term);

};

void
BlockInvariantFixpointQueue::updateSuccessors(
    FunctionValueTransitionMap& m,
    mlir::Block* src,
    const LocalsDomain& term) {

  // Iterate through successors.
  auto& smap = m.succMap(src);
  for (auto i = smap.begin(); i != smap.end(); ++i) {

    auto tgt = i->first;

    const auto& inherited = m.getInheritedValues(tgt);
    auto tgtDomainPtr = this->blockMap.find(tgt);
    if (tgtDomainPtr == this->blockMap.end()) {
      auto tgtArgInfo = addNew(tgt);
      ValueTranslator translator(inherited, *tgtArgInfo);
      tgtArgInfo->startDomain.populateFromPrev(translator, term);
    } else {
      auto tgtDomain = &tgtDomainPtr->second;
      ValueTranslator translator(inherited, *tgtDomain);
      if (tgtDomain->startDomain.mergeFromPrev(translator, term)) {
        pending.push_back(tgtDomain);
      }
    }
  }
}

void
BlockInvariantFixpointQueue::addSuccessors(
    FunctionValueTransitionMap& m,
    mlir::Block* src,
    mlir::OpBuilder& builder,
    const std::vector<mlir::Value> argValues,
    const LocalsDomain& term) {

  auto location = builder.getUnknownLoc();

  // Iterate through successors.
  auto& smap = m.succMap(src);
  for (auto i = smap.begin(); i != smap.end(); ++i) {
    auto tgt = i->first;
    auto rng = i->second;

    // Get list of arguments that successor needs.
    auto argIter = blockMap.find(tgt);
    if (argIter == blockMap.end()) {
      fatal_error("Could not find args for block.");
    }
    auto blockArgs = &argIter->second;
    // Iterate through arguments
    for (auto p : blockArgs->argVec) {
      // Skip if arg is no longer used.
      if (!p.first.scope)
        continue;

      // Lookup value to pass to block
      mlir::Value v =
        term.getScopeValue(p.first.scope, p.first.field,
          builder, location, argValues);

      // Add value to operand list.
      rng.append(v);
    }

  }
}

/// Here we utilize the CRTP `PassWrapper` utility class to provide some
/// necessary utility hooks. This is only necessary for passes defined directly
/// in C++. Passes defined declaratively use a cleaner mechanism for providing
/// these utilities.
class ScopeOptimization : public mlir::PassWrapper<ScopeOptimization,
                                           mlir::OperationPass<mlir::FuncOp>> {
private:
  void checkNoScopeOperands(mlir::Operation& op);
  void analyzeBlock(mlir::Block* block, LocalsDomain& locals);
  void optimizeBlock(mlir::Block* block, const std::vector<mlir::Value>& argValues, LocalsDomain& locals);


public:

  llvm::StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "resolveScope";
  }

  void runOnOperation() override;
};

// Check no operands are scope variables
void ScopeOptimization::checkNoScopeOperands(mlir::Operation& op) {
  const auto& operands = op.getOperands();
  auto scopeTypeId = mlir::python::ScopeType::getTypeID();

  for (auto i = operands.begin(); i != operands.end(); ++i) {
    mlir::Value v = *i;
    // Scopes must use known operations

    if (v.getType().getTypeID() == scopeTypeId) {
      return signalPassFailure();
    }
  }
}

void ScopeOptimization::analyzeBlock(mlir::Block* block, LocalsDomain& locals) {
  for (auto opPtr = block->begin(); opPtr != block->end(); ++opPtr) {
    auto opName = opPtr->getName().getIdentifier().str();
    if (auto derivedOp = mlir::dyn_cast<mlir::python::ScopeInit>(opPtr)) {
      locals.scope_init(derivedOp);
    } else if (auto derivedOp = mlir::dyn_cast<mlir::python::ScopeExtend>(opPtr)) {
      locals.scope_extend(derivedOp);
    } else if (auto derivedOp = mlir::dyn_cast<mlir::python::ScopeImport>(opPtr)) {
      locals.scope_import(derivedOp);
    } else if (auto op = mlir::dyn_cast<mlir::python::ScopeGet>(opPtr)) {
      locals.scope_get(op);
    } else if (auto op = mlir::dyn_cast<mlir::python::ScopeSet>(opPtr)) {
      locals.scope_domain(op.scope()).setValue(op.name(), op.value());
    } else {
      // Check no operands are scope variables
      checkNoScopeOperands(*opPtr);
    }
  }
}

void ScopeOptimization::optimizeBlock(
                      mlir::Block* block,
                      const std::vector<mlir::Value>& argValues,
                      LocalsDomain& locals) {
  std::vector<mlir::Operation*> toDelete;
  for (auto opPtr = block->begin(); opPtr != block->end(); ++opPtr) {
    auto opName = opPtr->getName().getIdentifier().str();
    if (auto derivedOp = mlir::dyn_cast<mlir::python::ScopeInit>(opPtr)) {
      locals.scope_init(derivedOp);
    } else if (auto derivedOp = mlir::dyn_cast<mlir::python::ScopeExtend>(opPtr)) {
      locals.scope_extend(derivedOp);
    } else if (auto derivedOp = mlir::dyn_cast<mlir::python::ScopeImport>(opPtr)) {
      locals.scope_import(derivedOp);
    } else if (auto op = mlir::dyn_cast<mlir::python::ScopeGet>(opPtr)) {
      if (auto d = locals.scope_get(op)) {
        mlir::OpBuilder builder(op.getContext());
        builder.setInsertionPoint(op);
        auto origResult = op.result();
        origResult.replaceAllUsesWith(d->getValue(builder, op.getLoc(), argValues));
        toDelete.push_back(op);
      }
    } else if (auto op = mlir::dyn_cast<mlir::python::ScopeSet>(opPtr)) {
      locals.scope_domain(op.scope()).setValue(op.name(), op.value());
    } else if (auto op = mlir::dyn_cast<mlir::python::FunctionRef>(opPtr)) {
      // FIXME: Collect invariants about closure passed into function.
    }
  }

  // Iterate and remove unused operators.
  for (auto op : toDelete) {
    op->erase();
  }
}

void ScopeOptimization::runOnOperation() {
  printf("Run on operation\n");
  // Get the current func::FuncOp operation being operated on.
  mlir::FuncOp fun = getOperation();

  auto ctx = &getContext();

  mlir::DominanceInfo& domInfo = getAnalysis<mlir::DominanceInfo>();
  FunctionValueTransitionMap fvtm(fun, domInfo);

  auto& blks = fun.getBlocks();

  if (blks.size() == 0) {
    return;
  }

  /// Build map from block to scopes they read to the values from each scope.
  llvm::DenseMap<llvm::StringRef, ValueDomain> map;

  BlockInvariantFixpointQueue inv(&*blks.begin());
  while (inv.hasNext()) {
    auto p = inv.nextBlock();
    mlir::Block* block = p->block;
    LocalsDomain blockScopes(p->startDomain);
    analyzeBlock(block, blockScopes);
    // Update all successor blocks.
    inv.updateSuccessors(fvtm, block, blockScopes);
  }

  // Run optimization passes
  auto pythonType = mlir::python::ValueType::get(ctx);
  mlir::Builder builder(fun.getContext());
  for (auto& p : inv.blockMap) {
    auto& argInfo = p.second;
    auto block = argInfo.block;
    // Add arguments for blocks
    std::vector<mlir::Value> argVec;
    for (auto p : argInfo.argVec) {
      if (p.first.scope) {
        auto arg = block->addArgument(pythonType, builder.getUnknownLoc());
        argVec.push_back(arg);
      } else {
        argVec.push_back(mlir::Value());
      }
    }
    // Optimize operations in block
    LocalsDomain blockScopes(argInfo.startDomain);
    optimizeBlock(block, argVec, blockScopes);

    mlir::OpBuilder builder(ctx);
    builder.setInsertionPoint(&block->back());

    // Add extra operands for successor blocks.
    inv.addSuccessors(fvtm, block, builder, argVec, blockScopes);
  }
}

}

int main(int argc, const char** argv) {
  if (argc != 2) {
    fprintf(stderr, "Please provide the path to read.\n");
    return -1;
  }
  const char* path = argv[1];


//  context.allowUnregisteredDialects();
  mlir::DialectRegistry registry;
  registry.insert<mlir::python::PythonDialect>();

  mlir::MLIRContext inputContext;
  initContext(inputContext, registry);
  mlir::Block block;
  mlir::LogicalResult r = mlir::parseSourceFile(path, &block, &inputContext);
  if (r.failed()) {
    fprintf(stderr, "Failed to parse mlir file file.\n");
    return -1;
  }
  if (block.getOperations().size() != 1) {
    fprintf(stderr, "Missing module.\n");
    return -1;
  }

  mlir::PassRegistration<ScopeOptimization>();

  // Create a top-level `PassManager` class. If an operation type is not
  // explicitly specific, the default is the builtin `module` operation.
  mlir::PassManager pm(&inputContext);
  auto &nestedFunctionPM = pm.nest<mlir::FuncOp>();
  nestedFunctionPM.addPass(std::make_unique<ScopeOptimization>());

  if (failed(pm.run(&*block.begin()))) {
    fprintf(stderr, "Pass failed\n");
    exit(-1);
  }

  block.begin()->dump();

  return 0;
}
