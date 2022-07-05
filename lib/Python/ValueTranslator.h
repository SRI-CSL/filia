#pragma once
#include "PythonDomain.h"
#include <vector>

struct ScopeField {
  mlir::Value scope;
  llvm::StringRef field;
};

using namespace llvm;

template <> struct llvm::DenseMapInfo<ScopeField, void> {
  static inline ScopeField getEmptyKey() {
    return { .scope = llvm::DenseMapInfo<mlir::Value, void>::getEmptyKey(),
             .field = llvm::DenseMapInfo<llvm::StringRef, void>::getEmptyKey() };
  }

  static inline ScopeField getTombstoneKey() {
    return { .scope = llvm::DenseMapInfo<mlir::Value, void>::getTombstoneKey(),
             .field = llvm::DenseMapInfo<llvm::StringRef, void>::getTombstoneKey() };
  }

  static unsigned getHashValue(const ScopeField& v) {
    return llvm::DenseMapInfo<mlir::Value, void>::getHashValue(v.scope)
         ^ llvm::DenseMapInfo<llvm::StringRef, void>::getHashValue(v.field);
  }

  static bool isEqual(const ScopeField& x, const ScopeField& y) {
    return llvm::DenseMapInfo<mlir::Value, void>::isEqual(x.scope, y.scope)
        && llvm::DenseMapInfo<llvm::StringRef, void>::isEqual(x.field, y.field);
  }
};

static inline
bool operator==(const ScopeField& x, const ScopeField& y) {
  return x.scope == y.scope && x.field == y.field;
}

/**
 * Stores information about potential additional arguments to create for blocks
 * for passing Python values directly instead of via cells.
 */
struct BlockArgInfo {
private:
public:
  LocalsDomain startDomain;


  // Map from scope fields to a unique index that identifies them.
  llvm::DenseMap<mlir::Value, unsigned> argMap;
  std::vector<std::pair<mlir::Value, std::vector<mlir::Location>> > argVec;

  BlockArgInfo(mlir::Block* b) : startDomain(b) { }

  BlockArgInfo() = delete;
  BlockArgInfo(BlockArgInfo&&) = default;
  BlockArgInfo(const BlockArgInfo&) = delete;
  BlockArgInfo& operator=(const BlockArgInfo&) = delete;

  mlir::Block* getBlock() const {
    return startDomain.getBlock();
  }

  // Returns an index to represent a value stored at a particular scope value.
  unsigned getLocalArg(const mlir::Value& cell) {
    auto i = argMap.find(cell);
    if (i != argMap.end())
      return i->second;

    unsigned r = argVec.size();
    argMap.insert(std::make_pair(cell, r));
    argVec.push_back(std::make_pair(cell, std::vector<mlir::Location>()));
    return r;
  }

  /**
   * Mark that a local argument is no longer needed.
   */
  void removeLocalArg(unsigned idx) {
    argVec[idx].first = mlir::Value();
  }
};

using InheritedSet = llvm::DenseSet<mlir::Value>;

/***
 * Provides capabilities for translating between domains.
 */
class ValueTranslator {
  const InheritedSet& inherited;

public:
  BlockArgInfo& argInfo;

  ValueTranslator(const InheritedSet& inherited, BlockArgInfo& argInfo)
    : inherited(inherited), argInfo(argInfo) {
  }

  // Translates a value from the source block into a value in the target block.
  mlir::Value valueToTarget(const mlir::Value& v) const {
    if (inherited.contains(v)) {
      return v;
    }
    return mlir::Value();
  }

  // Returns an index to represent a value stored at a particular scope value.
  unsigned getCellValueArg(const mlir::Value& cell) {
    return argInfo.getLocalArg(cell);
  }

  void removeLocalArg(unsigned idx) {
    argInfo.removeLocalArg(idx);
  }
};