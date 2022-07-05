#pragma once
#include <mlir/Parser.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/Pass.h>
#include <llvm/ADT/ImmutableMap.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallSet.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>

#include "Python/PythonDialect.h"
#include "Python/PythonOps.h"

class ValueTranslator;

/**
 *
 * Identifies information about a variable name stored in a scope.
 * Information is relative to a specific execution point in a block.
 *
 * There are currently three "types" of
 * * VALUE.  Indicates the name is associated with a specific value in the
 *   block.
 * * BUILTIN.  Indicates the name is associated with a builtin function.
 * * MODULE.  Indicates the name is associated with a specific module.
 * * ARGUMENT.  Indicates the name could be associated with an argument that
 *   could be added to the block.
 */
class ValueDomain {
public:
  enum ValueDomainType { VALUE, BUILTIN, MODULE, ARGUMENT };

  ValueDomainType type;
  // Value this is associated with.
  // (defined when type == VALUE)
  mlir::Value value;
  // Name of builtin or module
  // (defined when `type == BUILTIN || type == MODULE`).
  llvm::StringRef name;
  // Index of argument
  unsigned argument;

public:
  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(type);
    switch (type) {
      case VALUE:
        ID.AddPointer(value.getImpl());
        break;
      case BUILTIN:
        ID.AddString(name);
        break;
      case MODULE:
        ID.AddString(name);
        break;
      case ARGUMENT:
        ID.AddInteger(argument);
        break;
    }
  }

  static
  ValueDomain make_value(const mlir::Value& value) {
    return { .type = VALUE, .value = value };
  }

  static
  ValueDomain make_builtin(llvm::StringRef name) {
    return { .type = BUILTIN, .name = name };
  }

  static
  ValueDomain make_module(llvm::StringRef name) {
    return { .type = MODULE, .name = name };
  }

  static
  ValueDomain make_argument(unsigned arg) {
    return { .type = ARGUMENT, .argument = arg };
  }

  mlir::Value getValue(mlir::OpBuilder& builder,
                       const mlir::Location& location,
                       const std::vector<mlir::Value>& argValues) const {
    switch (type) {
    case VALUE:
      return value;
    case BUILTIN:
      {
        auto b = builder.create<mlir::python::Builtin>(location, name);
        return b.result();
      }
    case MODULE:
      {
        auto b = builder.create<mlir::python::Module>(location, name);
        return b.result();
      }
    case ARGUMENT:
      {
        auto r = argValues[argument];
        if (!r) llvm::report_fatal_error("Unassigned argValue");
        return r;
      }
    }
  }
};

static inline
bool operator==(const ValueDomain& x, const ValueDomain& y) {
  if (x.type != y.type)
    return false;
  switch (x.type) {
  case ValueDomain::VALUE:
    return x.value == y.value;
  case ValueDomain::BUILTIN:
    return x.name == y.name;
  case ValueDomain::MODULE:
    return x.name == y.name;
  case ValueDomain::ARGUMENT:
    return x.argument == y.argument;
  }
}

/// The contents of a cell.
class CellDomain {
public:

  enum Status { EMPTY, VALUE, CELL_UNKNOWN };
private:
  Status status;
  ValueDomain _value;

  CellDomain(Status status) {
    this->status = status;
  }

  CellDomain(Status status, ValueDomain&& d) {
    this->status = status;
    this->_value = d;
  }
public:

  Status getStatus() const { return this->status; }

  bool is_unknown(void) const {
    return status == CELL_UNKNOWN;
  }

  const ValueDomain& getValue() const {
    assert(status == VALUE);
    return _value;
  }

  static CellDomain unknown(void) {
    return CellDomain(CELL_UNKNOWN);
  }


  static CellDomain empty(void) {
    return CellDomain(EMPTY);
  }

  static CellDomain value(ValueDomain&& value) {
    return CellDomain(VALUE, std::move(value));
  }

  /// initializeFromPrev initializes a scope domain for a target block using information from the scope
  /// domain in a previous block.  This may need to requst the translator generate block
  /// arguments to pass values from the source to the target.
  ///
  /// @param translator Provides functionality for mapping values into target block
  /// @param srcDomain Source domain to pull constraints from.
  /// @param tgtCell MLIR value for cell in target block.
  ///
  static CellDomain initializeFromPrev(ValueTranslator& translator, const CellDomain& srcDomain, const mlir::Value& tgtCell);

  bool mergeFromPrev(ValueTranslator& translator, const CellDomain& srcDomain, const mlir::Value& tgtCell);
};

/**
 * This maps values in the block to the scope domain describing the invariants for
 * locals at a particular program location.
 */
class LocalsDomain {
  mlir::Block* block;
  // Map values for cell variables to information about the contents of the cell at this
  // scope variables to the associated domain.
  llvm::DenseMap<mlir::Value, CellDomain> cellValues;

  ValueDomain valueDomain(const mlir::Value& value) {
    // FIXME:
    return ValueDomain::make_value(value);
  }

public:
  LocalsDomain(mlir::Block* block) : block(block) {
      if (!block) llvm::report_fatal_error("Locals domain given null block.");
  }
  LocalsDomain(const LocalsDomain&) = default;
  LocalsDomain(LocalsDomain&&) = default;
  LocalsDomain& operator=(const LocalsDomain&) = delete;

  // Get block this domain is for.
  mlir::Block* getBlock(void) const { return block; }

  //!
  //! Create a locals domain from a previous block
  //!
  void populateFromPrev(ValueTranslator& translator, const LocalsDomain& prev);

  ///
  /// `x.mergeFromPrev(trans, prev) applies the translator and takes the
  /// domain containing facts true in both x and trans(prev).
  ///
  /// @return if the new domain information changes the domain.
  bool mergeFromPrev(ValueTranslator& translator, const LocalsDomain& prev);

  void cellUnknown(const mlir::Value& v) {
    cellValues.try_emplace(v, CellDomain::unknown());
  }

  /**
   * Update locals domain with cell allocation.
   *
   * @param op allocation operation.
   */
  void cellAlloc(mlir::python::CellAlloc op) {
    auto d
      = op.initial()
      ? CellDomain::value(this->valueDomain(op.initial()))
      : CellDomain::empty();
    cellValues.try_emplace(op.result(), d);
  }

  CellDomain& cellDomain(mlir::Value cell) {
    auto i = cellValues.find(cell);
    if (i == cellValues.end()) {
      std::string str;
      mlir::AsmState state(block->getParentOp());
      llvm::raw_string_ostream o(str);
      o << "Error in block ";
      block->printAsOperand(o);
      o << ": Have not seen cell ";
      cell.printAsOperand(o, state);
      o << ".";
      llvm::report_fatal_error(str.c_str());
    }
    return i->second;
  }

  const CellDomain& cellDomain(mlir::Value cell) const {
    auto i = cellValues.find(cell);
    if (i == cellValues.end()) {
      std::string str;
      mlir::AsmState state(block->getParentOp());
      llvm::raw_string_ostream o(str);
      o << "Error in block ";
      block->printAsOperand(o);
      o << ": Have not seen cell ";
      cell.printAsOperand(o, state);
      o << ".";
      llvm::report_fatal_error(str.c_str());
    }
    return i->second;
  }

  /**
   * Update locals domain with cell alloc
   *
   * @param op allocation
   */
  void cellStore(mlir::python::CellStore op) {
    auto& d = cellDomain(op.cell());
    d = CellDomain::value(this->valueDomain(op.value()));
  }

  mlir::Value cellValue(mlir::Value cell,
                            mlir::OpBuilder& builder,
                            const mlir::Location& location,
                            const std::vector<mlir::Value> &argValues) const {

    const auto& cd = cellDomain(cell);
    if (cd.getStatus() == CellDomain::VALUE) {
      return cd.getValue().getValue(builder, location, argValues);
    } else {
      return mlir::Value();
    }
  }

  mlir::Value cellLoad(mlir::python::CellLoad op, const std::vector<mlir::Value> &argValues) {
    const auto& cd = cellDomain(op.cell());
    if (cd.getStatus() == CellDomain::VALUE) {
      const auto& vd = cd.getValue();
      mlir::OpBuilder builder(op.getContext());
      builder.setInsertionPoint(op);
      auto v = vd.getValue(builder, op.getLoc(), argValues);
      auto b = builder.create<mlir::python::MkReturnOp>(op.getLoc(), v);
      return b.result();
    } else {
      return mlir::Value();
    }
  }
};