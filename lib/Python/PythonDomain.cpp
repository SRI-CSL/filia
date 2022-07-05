#include "PythonDomain.h"
#include "ValueTranslator.h"

using namespace llvm;

CellDomain CellDomain::initializeFromPrev(
    ValueTranslator& translator,
    const CellDomain& srcDomain,
    const mlir::Value& tgtCell) {
  switch (srcDomain.status) {
  case CellDomain::EMPTY:
    return CellDomain::empty();
  case CellDomain::CELL_UNKNOWN:
    return CellDomain::unknown();
  case CellDomain::VALUE:
    switch (srcDomain._value.type) {
    case ValueDomain::VALUE:
      if (auto newV = translator.valueToTarget(srcDomain._value.value)) {
        return CellDomain::value(ValueDomain::make_value(newV));
      } else {
        unsigned argIndex = translator.getCellValueArg(tgtCell);
        return CellDomain::value(ValueDomain::make_argument(argIndex));
      }
      break;
    case ValueDomain::BUILTIN:
    case ValueDomain::MODULE:
      return CellDomain::value(ValueDomain(srcDomain._value));
      break;
    case ValueDomain::ARGUMENT:
      {
        unsigned argIndex = translator.getCellValueArg(tgtCell);
        return CellDomain::value(ValueDomain::make_argument(argIndex));
      }
      break;
    default:
      report_fatal_error("Invalid value domain");
    }
  default:
    report_fatal_error("Invalid cell domain");
  }
}

bool CellDomain::mergeFromPrev(ValueTranslator& translator, const CellDomain& srcDomain, const mlir::Value& tgtCell) {
  switch (status) {
  case CellDomain::CELL_UNKNOWN:
    return false;
  case CellDomain::EMPTY:
    if (srcDomain.status != CellDomain::EMPTY) {
      status = CELL_UNKNOWN;
      return true;
    }
    return false;
  case CellDomain::VALUE:
    if (srcDomain.status != CellDomain::VALUE) {
      status = CELL_UNKNOWN;
      return true;
    } else {
      const auto& newDomain = srcDomain._value;
      if (_value.type == ValueDomain::ARGUMENT) {
        // Do nothing
        return false;
      } else if (_value == newDomain) {
        // Do nothing
        return false;
      } else {
        unsigned argIndex = translator.getCellValueArg(tgtCell);
        _value = ValueDomain::make_argument(argIndex);
        return true;
      }
    }
  }
}


/*
void ScopeDomain::addBuiltins() {
  addBuiltin(::mlir::python::BuiltinAttr::abs);
  addBuiltin(::mlir::python::BuiltinAttr::aiter);
  addBuiltin(::mlir::python::BuiltinAttr::all);
  addBuiltin(::mlir::python::BuiltinAttr::any);
  addBuiltin(::mlir::python::BuiltinAttr::anext);
  addBuiltin(::mlir::python::BuiltinAttr::ascii);
  addBuiltin(::mlir::python::BuiltinAttr::bin);
  addBuiltin(::mlir::python::BuiltinAttr::bool_builtin, "bool");
  addBuiltin(::mlir::python::BuiltinAttr::breakpoint);
  addBuiltin(::mlir::python::BuiltinAttr::bytearray);
  addBuiltin(::mlir::python::BuiltinAttr::bytes);
  addBuiltin(::mlir::python::BuiltinAttr::callable);
  addBuiltin(::mlir::python::BuiltinAttr::chr);
  addBuiltin(::mlir::python::BuiltinAttr::classmethod);
  addBuiltin(::mlir::python::BuiltinAttr::compile);
  addBuiltin(::mlir::python::BuiltinAttr::complex);
  addBuiltin(::mlir::python::BuiltinAttr::delattr);
  addBuiltin(::mlir::python::BuiltinAttr::dict);
  addBuiltin(::mlir::python::BuiltinAttr::dir);
  addBuiltin(::mlir::python::BuiltinAttr::divmod);
  addBuiltin(::mlir::python::BuiltinAttr::enumerate);
  addBuiltin(::mlir::python::BuiltinAttr::eval);
  addBuiltin(::mlir::python::BuiltinAttr::exec);
  addBuiltin(::mlir::python::BuiltinAttr::filter);
  addBuiltin(::mlir::python::BuiltinAttr::float_builtin, "float");
  addBuiltin(::mlir::python::BuiltinAttr::format);
  addBuiltin(::mlir::python::BuiltinAttr::frozenset);
  addBuiltin(::mlir::python::BuiltinAttr::getattr);
  addBuiltin(::mlir::python::BuiltinAttr::globals);
  addBuiltin(::mlir::python::BuiltinAttr::hasattr);
  addBuiltin(::mlir::python::BuiltinAttr::hash);
  addBuiltin(::mlir::python::BuiltinAttr::help);
  addBuiltin(::mlir::python::BuiltinAttr::hex);
  addBuiltin(::mlir::python::BuiltinAttr::id);
  addBuiltin(::mlir::python::BuiltinAttr::input);
  addBuiltin(::mlir::python::BuiltinAttr::int_builtin, "int");
  addBuiltin(::mlir::python::BuiltinAttr::isinstance);
  addBuiltin(::mlir::python::BuiltinAttr::issubclass);
  addBuiltin(::mlir::python::BuiltinAttr::iter);
  addBuiltin(::mlir::python::BuiltinAttr::len);
  addBuiltin(::mlir::python::BuiltinAttr::list);
  addBuiltin(::mlir::python::BuiltinAttr::locals);
  addBuiltin(::mlir::python::BuiltinAttr::map);
  addBuiltin(::mlir::python::BuiltinAttr::max);
  addBuiltin(::mlir::python::BuiltinAttr::memoryview);
  addBuiltin(::mlir::python::BuiltinAttr::min);
  addBuiltin(::mlir::python::BuiltinAttr::next);
  addBuiltin(::mlir::python::BuiltinAttr::object);
  addBuiltin(::mlir::python::BuiltinAttr::oct);
  addBuiltin(::mlir::python::BuiltinAttr::open);
  addBuiltin(::mlir::python::BuiltinAttr::ord);
  addBuiltin(::mlir::python::BuiltinAttr::pow);
  addBuiltin(::mlir::python::BuiltinAttr::print);
  addBuiltin(::mlir::python::BuiltinAttr::property);
  addBuiltin(::mlir::python::BuiltinAttr::range);
  addBuiltin(::mlir::python::BuiltinAttr::repr);
  addBuiltin(::mlir::python::BuiltinAttr::reversed);
  addBuiltin(::mlir::python::BuiltinAttr::round);
  addBuiltin(::mlir::python::BuiltinAttr::set);
  addBuiltin(::mlir::python::BuiltinAttr::setattr);
  addBuiltin(::mlir::python::BuiltinAttr::slice);
  addBuiltin(::mlir::python::BuiltinAttr::sorted);
  addBuiltin(::mlir::python::BuiltinAttr::staticmethod);
  addBuiltin(::mlir::python::BuiltinAttr::str);
  addBuiltin(::mlir::python::BuiltinAttr::sum);
  addBuiltin(::mlir::python::BuiltinAttr::super);
  addBuiltin(::mlir::python::BuiltinAttr::tuple);
  addBuiltin(::mlir::python::BuiltinAttr::type);
  addBuiltin(::mlir::python::BuiltinAttr::vars);
  addBuiltin(::mlir::python::BuiltinAttr::zip);
  addBuiltin(::mlir::python::BuiltinAttr::import, "__import__");
}
*/

// Create a locals domain from a previous block
void LocalsDomain::populateFromPrev(ValueTranslator& translator, const LocalsDomain& prev) {
  // Iterate through all values in previous block.
  for (auto i = prev.cellValues.begin(); i != prev.cellValues.end(); ++i) {
    auto cell = i->first;
    auto& srcDomain = i->second;
    auto tgtCell = translator.valueToTarget(cell);
    if (!tgtCell) {
      std::string str;
      mlir::AsmState state(block->getParentOp());
      llvm::raw_string_ostream o(str);
      o << "Error in block ";
      block->printAsOperand(o);
      o << ": Dropping cell ";
      cell.printAsOperand(o, state);
      o << ".";
      report_fatal_error(str.c_str());

//      continue;

    }
    auto p = cellValues.try_emplace(tgtCell,
      CellDomain::initializeFromPrev(translator, srcDomain, tgtCell));
    if (!p.second) {
      report_fatal_error("Translator maps two values to single value.");
    }
  }
}

/**
 * `x.mergeFromPrev(trans, prev) applies the translator and takes the
 * domain containing facts true in both x and trans(prev).
 */
bool LocalsDomain::mergeFromPrev(ValueTranslator& translator, const LocalsDomain& prev) {
  // Create set containing all the cells in this block that have associated domains.
  llvm::DenseSet<mlir::Value> seen;

  // Propagate each constraint in previous block to this  block.
  bool changed = false;
  for (auto i = prev.cellValues.begin(); i != prev.cellValues.end(); ++i) {
    auto prevCell = i->first;
    auto& srcDomain = i->second;
    auto tgtCell = translator.valueToTarget(prevCell);
    if (!tgtCell)
      continue;
    // Lookup domain for target value in this set.
    auto prevTgtIter = this->cellValues.find(tgtCell);
    if (prevTgtIter == this->cellValues.end())
      continue;
    if (!seen.insert(tgtCell).second) {
      report_fatal_error("Duplicate values in scope.");
    }
    if (prevTgtIter->second.mergeFromPrev(translator, srcDomain, tgtCell))
      changed = true;
  }


  // Remove all cells that were unconstrained in prev (and thus still in unseen)
  for (auto i = cellValues.begin(); i != cellValues.end(); ++i) {
    if (!i->second.is_unknown() && !seen.contains(i->first)) {
      i->second = CellDomain::unknown();
      changed = true;
    }
  }

  return changed;
}
