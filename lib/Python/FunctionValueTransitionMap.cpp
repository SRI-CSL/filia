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

FunctionValueTransitionMap::FunctionValueTransitionMap(mlir::FuncOp fun, mlir::DominanceInfo& domInfo) {

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
