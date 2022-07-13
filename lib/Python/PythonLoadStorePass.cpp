#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

#include "Python/PythonLoadStorePass.h"
#include "FunctionValueTransitionMap.h"
#include "ValueTranslator.h"

using namespace llvm;

namespace {

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
      report_fatal_error("nextBlock called when pending is empty.");
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
    report_fatal_error("Could not find args for block.");
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

using TermSubstFn = std::function<void(
  BlockInvariantFixpointQueue& inv,
  std::vector<mlir::Operation*>& toDelete,
  const std::vector<mlir::Value>& argValues,
  const LocalsDomain& term,
  mlir::Operation* opPtr)>;


using TermSubstFnMap = llvm::DenseMap<llvm::StringRef, TermSubstFn>;

template<typename T>
static void addOp(TermSubstFnMap& m) {
  m.try_emplace(T::getOperationName(), &addOpArgs<T>);
}


/**
 * Create map from supported terminal function names to the code for optimizng them.
 *
 */
TermSubstFnMap mkTermMap(void) {

  TermSubstFnMap termFns;
  addOp<mlir::cf::BranchOp>(termFns);
  addOp<mlir::cf::CondBranchOp>(termFns);
  addOp<mlir::func::ReturnOp>(termFns);
  addOp<mlir::python::RetBranchOp>(termFns);
  return termFns;
}

llvm::DenseMap<llvm::StringRef, TermSubstFn> termFns = mkTermMap();

}

namespace mlir {
namespace python {

/**
 * This function updates \p locals by applying the operations in the block.
 *
 * @param locals Invariants on block
 * @param block Block to anlyze
 */
static
void applyBlockOps(LocalsDomain& locals, mlir::Block* block) {
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

static
void optimizeBlock(BlockInvariantFixpointQueue& inv,
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
    report_fatal_error(str.c_str());
  }
}

static
void optimizeFunction(mlir::MLIRContext* ctx,
                      mlir::DominanceInfo& domInfo,
                      mlir::func::FuncOp fun) {

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

class PythonLoadStoreOptimization
   : public PassWrapper<PythonLoadStoreOptimization, OperationPass<mlir::ModuleOp>> {
public:

  llvm::StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "python-load-store";
  }

  void runOnOperation() override;
};

void PythonLoadStoreOptimization::runOnOperation() {
  auto ctx = &this->getContext();
  mlir::DominanceInfo& domInfo = getAnalysis<mlir::DominanceInfo>();

  // Get the current func::FuncOp operation being operated on.
  mlir::ModuleOp m = getOperation();
  ::mlir::Region& r = m.getBodyRegion();
  for (auto iBlock = r.begin(); iBlock != r.end(); ++iBlock) {
    for (auto iOp = iBlock->begin(); iOp != iBlock->end(); ++iOp) {
      auto& op = *iOp;
      if (auto funOp = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
        optimizeFunction(ctx, domInfo, funOp);
      }
    }
  }
}

void registerLoadStorePass() {
  mlir::PassRegistration<mlir::python::PythonLoadStoreOptimization>();
}


}
}