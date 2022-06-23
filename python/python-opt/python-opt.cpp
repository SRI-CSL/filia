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
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
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
  for (auto& blk : body->getBlocks()) {
    llvm::DenseSet<mlir::Value> s;
    for (auto arg : blk.getArguments())
      s.insert(arg);
    for (auto& op : blk) {
      s.insert(op.getResults().begin(), op.getResults().end());
    }
    definedValues.try_emplace(&blk, std::move(s));
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
    auto p = inheritedValues.try_emplace(&*blk);
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
      if (add_call_edges<mlir::python::RetBranchOp>(s, opPtr)) {
      } else if (add_call_edges<mlir::python::InvokeOp>(s, opPtr)) {
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
      } else if (auto op = mlir::dyn_cast<mlir::cf::BranchOp>(opPtr)) {
        addSuccessorEdge(s, 0, op.getDest(), op.getDestOperandsMutable());
      } else if (auto op = mlir::dyn_cast<mlir::cf::CondBranchOp>(opPtr)) {
        addSuccessorEdge(s, 0, op.getTrueDest(),  op.getTrueDestOperandsMutable());
        addSuccessorEdge(s, 0, op.getFalseDest(), op.getFalseDestOperandsMutable());
//      } else if (mlir::isa<mlir::python::ThrowOp>(opPtr)) {
        // Do nothing on throw
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

  std::vector<mlir::Block*> pending;

  void addPending(mlir::Block* argInfo) {
    pending.push_back(argInfo);
  }
public:
  BlockInvariantFixpointQueue(mlir::Block* entry) {
    assert(entry != 0);
    auto i = blockMap.try_emplace(entry, entry).first;
    auto argInfo = &i->second;
    addPending(entry);
    // Declare scopes for argments.
    auto cellTypeId = mlir::python::CellType::getTypeID();
    for (auto a : entry->getArguments()) {
      if (a.getType().getTypeID() == cellTypeId)
        argInfo->startDomain.cellUnknown(a);
    }
  }

  BlockInvariantFixpointQueue() = delete;
  BlockInvariantFixpointQueue(const BlockInvariantFixpointQueue&) = delete;
  BlockInvariantFixpointQueue& operator=(const BlockInvariantFixpointQueue&) = delete;

  /**
   * Return true if there is another block to process.
   */
  bool hasNext() {
    return !pending.empty();
  }


  /**
   * Get next block and initial abstract state.
   */
  BlockArgInfo& nextBlock() {
    if (pending.empty()) {
      fatal_error("nextBlock called when pending is empty.");
    }
    auto p = pending.back();
    pending.pop_back();
    return blockMap.find(p)->second;
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

  void addSuccessors(
      std::vector<mlir::Value>& rng,
      mlir::Block* tgt,
      mlir::Operation::operand_range tgtArgs,
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
    assert(tgt != 0);

    const auto& inherited = m.getInheritedValues(tgt);
    auto tgtDomainPtr = blockMap.find(tgt);
    if (tgtDomainPtr == blockMap.end()) {
      auto i = blockMap.try_emplace(tgt, tgt).first;
      ValueTranslator translator(inherited, i->second);
      i->second.startDomain.populateFromPrev(translator, term);
      addPending(tgt);
    } else {
      auto& tgtDomain = tgtDomainPtr->second;
      ValueTranslator translator(inherited, tgtDomain);
      if (tgtDomain.startDomain.mergeFromPrev(translator, term)) {
        addPending(tgtDomain.getBlock());
      }
    }
  }
}

void
BlockInvariantFixpointQueue::addSuccessors(
    std::vector<mlir::Value>& rng,
    mlir::Block* tgt,
    mlir::Operation::operand_range tgtArgs,
    mlir::OpBuilder& builder,
    const std::vector<mlir::Value> argValues,
    const LocalsDomain& term) {

  auto location = builder.getUnknownLoc();

  // Get list of arguments that successor needs.
  auto argIter = blockMap.find(tgt);
  if (argIter == blockMap.end()) {
    fatal_error("Could not find args for block.");
  }

  for (auto a : tgtArgs) {
    rng.push_back(a);
  }

  auto blockArgs = &argIter->second;
  // Iterate through arguments
  for (auto p : blockArgs->argVec) {
    // Skip if arg is no longer used.
    if (!p.first)
      continue;

    // Lookup value to pass to block
    mlir::Value v = term.cellValue(p.first, builder, location, argValues);

    // Add value to operand list.
    rng.push_back(v);
  }
}

/*
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
*/

class ScopeOptimization : public mlir::PassWrapper<ScopeOptimization,
                                           mlir::OperationPass<mlir::ModuleOp>> {
private:
  void checkNoScopeOperands(mlir::Operation& op);

  /**
   * This function updates \p locals by applying the operations in the block.
   *
   * @param locals Invariants on block
   * @param block Block to anlyze
   */
  void applyBlockOps(LocalsDomain& locals, mlir::Block* block);
  void optimizeBlock(BlockInvariantFixpointQueue& inv, mlir::Block* block, const std::vector<mlir::Value>& argValues, LocalsDomain& locals);
public:

  llvm::StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "resolveScope";
  }

  void optimizeFunction(mlir::FuncOp fun);

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
      llvm::errs() << "Unsupported operand: ";
      op.getName().print(llvm::errs());
      llvm::errs() << "\n";
      fatal_error("Scope operand!");
//      return signalPassFailure();
    }
  }
}

void ScopeOptimization::applyBlockOps(LocalsDomain& locals, mlir::Block* block) {
  for (auto opPtr = block->begin(); opPtr != block->end(); ++opPtr) {
    if (auto derivedOp = mlir::dyn_cast<mlir::python::CellAlloc>(opPtr)) {
      locals.cellAlloc(derivedOp);
    } else if (auto derivedOp = mlir::dyn_cast<mlir::python::CellStore>(opPtr)) {
      locals.cellStore(derivedOp);
    } else if (auto op = mlir::dyn_cast<mlir::python::CellLoad>(opPtr)) {
      // Do nothing
    }
  }
}

using TermSubstFn = std::function<void(
  BlockInvariantFixpointQueue& inv,
  std::vector<mlir::Operation*>& toDelete,
  const std::vector<mlir::Value>& argValues,
  const LocalsDomain& term,
  mlir::Operation* opPtr)>;

template<typename T>
void addUnaryOpArgs(BlockInvariantFixpointQueue& inv,
                  std::vector<mlir::Operation*>& toDelete,
                  const std::vector<mlir::Value>& argValues,
                  const LocalsDomain& term,
                  mlir::Operation* opPtr) {
  assert(mlir::isa<T>(opPtr));
  auto op = mlir::cast<T>(opPtr);
  auto builder(mlir::OpBuilder::atBlockEnd(opPtr->getBlock()));

  std::vector<mlir::Value> returnOps;
  inv.addSuccessors(returnOps, op.returnDest(), op.returnDestOperands(), builder, argValues, term);

  std::vector<mlir::Value> exceptOps;
  inv.addSuccessors(exceptOps, op.exceptDest(), op.exceptDestOperands(), builder, argValues, term);

  builder.create<T>(op.getLoc(), op.arg(),
    returnOps, exceptOps, op.returnDest(), op.exceptDest());
  opPtr->erase();
}

template<typename T>
void addBinOpArgs(BlockInvariantFixpointQueue& inv,
                  std::vector<mlir::Operation*>& toDelete,
                  const std::vector<mlir::Value>& argValues,
                  const LocalsDomain& term,
                  mlir::Operation* opPtr) {
  assert(mlir::isa<T>(opPtr));

  auto op = mlir::cast<T>(opPtr);
  auto builder(mlir::OpBuilder::atBlockEnd(opPtr->getBlock()));

  std::vector<mlir::Value> returnOps;
  inv.addSuccessors(returnOps, op.returnDest(), op.returnDestOperands(), builder, argValues, term);

  std::vector<mlir::Value> exceptOps;
  inv.addSuccessors(exceptOps, op.exceptDest(), op.exceptDestOperands(), builder, argValues, term);

  builder.create<T>(op.getLoc(), op.lhs(), op.rhs(),
    returnOps, exceptOps, op.returnDest(), op.exceptDest());
  opPtr->erase();
}

template<typename T>
void addOpArgs(BlockInvariantFixpointQueue& inv,
               std::vector<mlir::Operation*>& toDelete,
               const std::vector<mlir::Value>& argValues,
                  const LocalsDomain& term,
                  mlir::Operation* opPtr);


template<>
void addOpArgs<mlir::cf::BranchOp>(BlockInvariantFixpointQueue& inv,
                  std::vector<mlir::Operation*>& toDelete,
                  const std::vector<mlir::Value>& argValues,
                  const LocalsDomain& term,
                  mlir::Operation* opPtr) {
  auto op = mlir::dyn_cast<mlir::cf::BranchOp>(opPtr);
  assert(op);

  auto builder(mlir::OpBuilder::atBlockEnd(opPtr->getBlock()));

  std::vector<mlir::Value> destOps;
  inv.addSuccessors(destOps, op.getDest(), op.getDestOperands(), builder, argValues, term);

  builder.create<mlir::cf::BranchOp>(op.getLoc(), destOps, op.getDest());
  opPtr->erase();
}

template<>
void addOpArgs<mlir::cf::CondBranchOp>(
        BlockInvariantFixpointQueue& inv,
        std::vector<mlir::Operation*>& toDelete,
        const std::vector<mlir::Value>& argValues,
        const LocalsDomain& term,
        mlir::Operation* opPtr) {
  auto op = mlir::dyn_cast<mlir::cf::CondBranchOp>(opPtr);
  assert (op);

  auto builder(mlir::OpBuilder::atBlockEnd(opPtr->getBlock()));

  std::vector<mlir::Value> trueOps;
  inv.addSuccessors(trueOps, op.getTrueDest(), op.getTrueDestOperands(), builder, argValues, term);

  std::vector<mlir::Value> falseOps;
  inv.addSuccessors(falseOps, op.getFalseDest(), op.getFalseDestOperands(), builder, argValues, term);

  builder.create<mlir::cf::CondBranchOp>(op.getLoc(), op.getCondition(), trueOps, falseOps,
    op.getTrueDest(), op.getFalseDest());
  opPtr->erase();
}

template<>
void addOpArgs<mlir::python::InvokeOp>(
        BlockInvariantFixpointQueue& inv,
        std::vector<mlir::Operation*>& toDelete,
        const std::vector<mlir::Value>& argValues,
        const LocalsDomain& term,
        mlir::Operation* opPtr) {
  auto op = mlir::dyn_cast<mlir::python::InvokeOp>(opPtr);
  assert(op);

  auto builder(mlir::OpBuilder::atBlockEnd(opPtr->getBlock()));

  std::vector<mlir::Value> returnOps;
  inv.addSuccessors(returnOps, op.returnDest(), op.returnDestOperands(), builder, argValues, term);

  std::vector<mlir::Value> exceptOps;
  inv.addSuccessors(exceptOps, op.exceptDest(), op.exceptDestOperands(), builder, argValues, term);

  mlir::ArrayAttr keywords = op.keywords() ? op.keywords().getValue() : mlir::ArrayAttr();

  builder.create<mlir::python::InvokeOp>(op.getLoc(), op.callee(), op.args(),
    keywords, returnOps, exceptOps, op.returnDest(), op.exceptDest());
  opPtr->erase();
}

template<>
void addOpArgs<mlir::python::ThrowOp>(
        BlockInvariantFixpointQueue& inv,
        std::vector<mlir::Operation*>& toDelete,
        const std::vector<mlir::Value>& argValues,
        const LocalsDomain& term,
        mlir::Operation* opPtr) {
  assert(mlir::isa<mlir::python::ThrowOp>(opPtr));
}

template<>
void addOpArgs<mlir::func::ReturnOp>(
        BlockInvariantFixpointQueue& inv,
        std::vector<mlir::Operation*>& toDelete,
        const std::vector<mlir::Value>& argValues,
        const LocalsDomain& term,
        mlir::Operation* opPtr) {
  assert(mlir::isa<mlir::func::ReturnOp>(opPtr));
}

template<>
void addOpArgs<mlir::python::RetBranchOp>(
        BlockInvariantFixpointQueue& inv,
        std::vector<mlir::Operation*>& toDelete,
        const std::vector<mlir::Value>& argValues,
        const LocalsDomain& term,
        mlir::Operation* opPtr) {
  assert(mlir::isa<mlir::python::RetBranchOp>(opPtr));

  auto op = mlir::cast<mlir::python::RetBranchOp>(opPtr);
  auto builder(mlir::OpBuilder::atBlockEnd(opPtr->getBlock()));

  std::vector<mlir::Value> returnOps;
  inv.addSuccessors(returnOps, op.returnDest(), op.returnDestOperands(), builder, argValues, term);

  std::vector<mlir::Value> exceptOps;
  inv.addSuccessors(exceptOps, op.exceptDest(), op.exceptDestOperands(), builder, argValues, term);

  builder.create<mlir::python::RetBranchOp>(op.getLoc(), op.value(),
    returnOps, exceptOps, op.returnDest(), op.exceptDest());
  opPtr->erase();
}

using TermSubstFnMap = llvm::DenseMap<llvm::StringRef, TermSubstFn>;

template<typename T>
static void addUnaryOp(TermSubstFnMap& m) {
  m.try_emplace(T::getOperationName(), &addUnaryOpArgs<T>);
}

template<typename T>
static void addBinOp(TermSubstFnMap& m) {
  m.try_emplace(T::getOperationName(), &addBinOpArgs<T>);
}

template<typename T>
static void addOp(TermSubstFnMap& m) {
  m.try_emplace(T::getOperationName(), &addOpArgs<T>);
}

/**
 * Create map from supported terminal function names to the code for optimizng them.
 *
 */
llvm::DenseMap<llvm::StringRef, TermSubstFn> mkTermMap(void) {

  llvm::DenseMap<llvm::StringRef, TermSubstFn> termFns;
  addOp<mlir::cf::BranchOp>(termFns);
  addOp<mlir::cf::CondBranchOp>(termFns);
  addOp<mlir::func::ReturnOp>(termFns);
  addOp<mlir::python::InvokeOp>(termFns);
  addOp<mlir::python::RetBranchOp>(termFns);
  addOp<mlir::python::ThrowOp>(termFns);
  addUnaryOp<mlir::python::InvertOp>(termFns);
  addBinOp<mlir::python::AddOp>(termFns);
  addBinOp<mlir::python::BitAndOp>(termFns);
  addBinOp<mlir::python::BitOrOp>(termFns);
  addBinOp<mlir::python::BitXorOp>(termFns);
  addBinOp<mlir::python::DivOp>(termFns);
  addBinOp<mlir::python::FloorDivOp>(termFns);
  addBinOp<mlir::python::LShiftOp>(termFns);
  addBinOp<mlir::python::ModOp>(termFns);
  addBinOp<mlir::python::MultOp>(termFns);
  addBinOp<mlir::python::MatMultOp>(termFns);
  addBinOp<mlir::python::PowOp>(termFns);
  addBinOp<mlir::python::RShiftOp>(termFns);
  addBinOp<mlir::python::SubOp>(termFns);
  addBinOp<mlir::python::EqOp>(termFns);
  addBinOp<mlir::python::GtOp>(termFns);
  addBinOp<mlir::python::GtEOp>(termFns);
  addBinOp<mlir::python::InOp>(termFns);
  addBinOp<mlir::python::IsOp>(termFns);
  addBinOp<mlir::python::IsNotOp>(termFns);
  addBinOp<mlir::python::LtOp>(termFns);
  addBinOp<mlir::python::LtEOp>(termFns);
  addBinOp<mlir::python::NotEqOp>(termFns);
  addBinOp<mlir::python::NotInOp>(termFns);
  return termFns;
}

llvm::DenseMap<llvm::StringRef, TermSubstFn> termFns = mkTermMap();

void ScopeOptimization::optimizeBlock(
                      BlockInvariantFixpointQueue& inv,
                      mlir::Block* block,
                      const std::vector<mlir::Value>& argValues,
                      LocalsDomain& locals) {

  std::vector<mlir::Operation*> toDelete;
  bool doSucc = true;
  for (auto opPtr = block->begin(); opPtr != block->end(); ++opPtr) {
    if (auto derivedOp = mlir::dyn_cast<mlir::python::CellAlloc>(opPtr)) {
      locals.cellAlloc(derivedOp);
    } else if (auto derivedOp = mlir::dyn_cast<mlir::python::CellStore>(opPtr)) {
      locals.cellStore(derivedOp);
    } else if (auto op = mlir::dyn_cast<mlir::python::CellLoad>(opPtr)) {
      if (auto v = locals.cellLoad(op, argValues)) {
        op.result().replaceAllUsesWith(v);
        toDelete.push_back(op);
      }
    } else {
      auto i = termFns.find(opPtr->getName().getStringRef());
      if (i != termFns.end()) {
        i->second(inv, toDelete, argValues, locals, &*opPtr);
        doSucc = false;
        break;
      }
    }
  }

  // Iterate and remove unused operators.
  for (auto op : toDelete) {
    op->erase();
  }
  if (doSucc) {
    std::string str;
    llvm::raw_string_ostream o(str);
    o << "Missing support for terminal instruction " << block->back() << ".";
    fatal_error(str.c_str());
  }
}

void ScopeOptimization::optimizeFunction(mlir::FuncOp fun) {

  mlir::DominanceInfo& domInfo = getAnalysis<mlir::DominanceInfo>();
  FunctionValueTransitionMap fvtm(fun, domInfo);

  auto& blks = fun.getBlocks();

  if (blks.size() == 0) {
    return;
  }

  BlockInvariantFixpointQueue inv(&*blks.begin());
  while (inv.hasNext()) {
    auto& p = inv.nextBlock();
    mlir::Block* block = p.getBlock();
    LocalsDomain locals(p.startDomain);
    applyBlockOps(locals, block);
    // Update all successor blocks.
    inv.updateSuccessors(fvtm, block, locals);
  }

  auto ctx = &this->getContext();

  // Run optimization passes
  auto pythonValueType = mlir::python::ValueType::get(ctx);
  mlir::Builder builder(fun.getContext());
  for (auto& p : inv.blockMap) {
    auto& argInfo = p.second;
    auto block = argInfo.getBlock();

    // Add value arguments for blocks identified by cells.
    std::vector<mlir::Value> argVec;
    for (auto p : argInfo.argVec) {
      if (p.first) {
        auto arg = block->addArgument(pythonValueType, builder.getUnknownLoc());
        argVec.push_back(arg);
      } else {
        argVec.push_back(mlir::Value());
      }
    }
    // Optimize operations in block
    LocalsDomain blockLocals(argInfo.startDomain);
    optimizeBlock(inv, block, argVec, blockLocals);
  }
}


void ScopeOptimization::runOnOperation() {
  // Get the current func::FuncOp operation being operated on.
  mlir::ModuleOp m = getOperation();
  auto& r = m.body();
  for (auto iBlock = r.begin(); iBlock != r.end(); ++iBlock) {
    for (auto iOp = iBlock->begin(); iOp != iBlock->end(); ++iOp) {
      auto& op = *iOp;
      if (auto funOp = mlir::dyn_cast<mlir::FuncOp>(op)) {
        optimizeFunction(funOp);
      }
    }
  }
}

}

int main(int argc, const char** argv) {
  const char* path = 0;
  bool verify = false;

  for (int i = 1; i != argc; ++i) {
    if (strcmp(argv[i], "--verify") == 0) {
      verify = true;
    } else {
      if (path != 0) {
        fprintf(stderr, "Please specify only a single file to read.");
        exit(-1);
      }
      path = argv[i];
    }
  }

  if (!path) {
    fprintf(stderr, "Please provide the path to read.\n");
    exit(-1);
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::python::PythonDialect>();

  mlir::MLIRContext ctx;
  initContext(ctx, registry);
  mlir::Block block;
  mlir::LogicalResult r = mlir::parseSourceFile(path, &block, &ctx);
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
  mlir::PassManager pm(&ctx);
  pm.enableVerifier(verify);
//  auto &nestedFunctionPM = pm.nest<mlir::ModuleOp>();
//  nestedFunctionPM.addPass(std::make_unique<ScopeOptimization>());
  pm.addPass(std::make_unique<ScopeOptimization>());
//  pm.addPass()

  if (failed(pm.run(&*block.begin()))) {
    fprintf(stderr, "Pass failed\n");
    exit(-1);
  }

  block.begin()->print(llvm::outs());
  llvm::outs() << "\n";

  return 0;
}
