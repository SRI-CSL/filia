#pragma once
#include "PythonDomain.h"
#include <vector>

struct ScopeField {
  mlir::Value scope;
  llvm::StringRef field;
};

using namespace llvm;

template <> struct DenseMapInfo<ScopeField, void> {
  static inline ScopeField getEmptyKey() {
    return { .scope = DenseMapInfo<mlir::Value, void>::getEmptyKey(), .field = DenseMapInfo<llvm::StringRef, void>::getEmptyKey() };
  }

  static inline ScopeField getTombstoneKey() {
    return { .scope = DenseMapInfo<mlir::Value, void>::getTombstoneKey(), .field = DenseMapInfo<llvm::StringRef, void>::getTombstoneKey() };
  }

  static unsigned getHashValue(const ScopeField& v) {
    return DenseMapInfo<mlir::Value, void>::getHashValue(v.scope) ^ DenseMapInfo<llvm::StringRef, void>::getHashValue(v.field);
  }

  static bool isEqual(const ScopeField& x, const ScopeField& y) {
    return DenseMapInfo<mlir::Value, void>::isEqual(x.scope, y.scope)
        && DenseMapInfo<llvm::StringRef, void>::isEqual(x.field, y.field);
  }
};

static inline
bool operator==(const ScopeField& x, const ScopeField& y) {
  return x.scope == y.scope && x.field == y.field;
}

/**
 * Store additional arguments to create for blocks for passing Python values
 * directly instead of via scopes.
 *
 */
struct BlockArgInfo {
private:
  // Block this arg info is for..
  mlir::Block* block;
public:
  LocalsDomain startDomain;


  // Map from scope fields to a unique index that identifies them.
  llvm::DenseMap<ScopeField, unsigned> argMap;
  std::vector<std::pair<ScopeField, std::vector<mlir::Location>> > argVec;

  BlockArgInfo(mlir::Block* b) : block(b) {
    if (!b) fatal_error("BlockArgInfo given null block.");
  }

  BlockArgInfo() = delete;
  BlockArgInfo(BlockArgInfo&&) = default;
  BlockArgInfo(const BlockArgInfo&) = delete;
  BlockArgInfo& operator=(const BlockArgInfo&) = delete;

  mlir::Block* getBlock() const {
    //if (!block) {
    //  fatal_error("getBlock() is null.");
    //}
    return block;
  }

  // Returns an index to represent a value stored at a particular scope value.
  unsigned getLocalArg(const mlir::Value& scope, const llvm::StringRef name) {
    ScopeField sf = { .scope = scope, .field = name };
    auto i = argMap.find(sf);
    if (i != argMap.end())
      return i->second;

    unsigned r = argVec.size();
    argMap.insert(std::make_pair(sf, r));
    argVec.push_back(std::make_pair(sf, std::vector<mlir::Location>()));
    return r;
  }

  /**
   * Mark that a local argument is no longer needed.
   */
  void removeLocalArg(unsigned idx) {
    argVec[idx].first.scope = mlir::Value();
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
  unsigned getLocalArg(const mlir::Value& scope, const llvm::StringRef name) {
    return argInfo.getLocalArg(scope, name);
  }

  void removeLocalArg(unsigned idx) {
    argInfo.removeLocalArg(idx);
  }
};