#include "FunctionValueTransitionMap.h"
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

/**
 * Walk through all operations in a list of blocks to populate map from blocks to defined
 * values.
 *
 * @param definedValues Map to populate
 * @param blks Blocks in a function
 */
static
void populateDefinedValues(FunctionValueTransitionMap::BlockValueMap& definedValues, mlir::Region* body) {
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
void populateInheritedValues(mlir::DominanceInfo& domInfo,
                             FunctionValueTransitionMap::BlockValueMap& inheritedValues,
                             const FunctionValueTransitionMap::BlockValueMap& definedValues,
                             mlir::Region* body) {
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

FunctionValueTransitionMap::FunctionValueTransitionMap(mlir::func::FuncOp fun, mlir::DominanceInfo& domInfo) {

  auto body = &fun.getBody();
  populateDefinedValues(definedValues, body);
  populateInheritedValues(domInfo, inheritedValues, definedValues, body);

  auto& blks = body->getBlocks();
  // Populate inherited values given defined values
  for (auto blk = blks.begin(); blk != blks.end(); ++blk) {
    auto& s = transMap.insert(std::make_pair(&*blk, SuccessorMap())).first->second;

    auto opPtr = &blk->back();
    if (add_call_edges<mlir::python::RetBranchOp>(s, opPtr)) {
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
