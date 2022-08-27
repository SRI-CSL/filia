#pragma once

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <vector>

#include "ValueTranslator.h"

namespace mlir {
class Block;
class Value;
class MutableOperandRange;
class DominanceInfo;
}

/**
 * This maps how values in one block are translated into a successor.
 */
class FunctionValueTransitionMap {
public:
  using BlockValueMap = llvm::DenseMap<mlir::Block*, llvm::DenseSet<mlir::Value>>;

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
  FunctionValueTransitionMap(mlir::func::FuncOp fun, mlir::DominanceInfo& domInfo);

  const InheritedSet& getInheritedValues(mlir::Block* block) {
    auto i = inheritedValues.find(block);
    if (i == inheritedValues.end()) {
      llvm::report_fatal_error("Could not find block.");
    }
    return i->second;
  }

  const SuccessorMap& succMap(mlir::Block* block) {
    auto i = transMap.find(block);
    if (i == transMap.end()) {
      llvm::report_fatal_error("Could not find block.");
    }
    return i->second;
  }
};