#include "PythonDomain.h"
#include "ValueTranslator.h"

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

void ScopeDomain::initializeFromPrev(ValueTranslator& translator, const ScopeDomain& srcDomain, const mlir::Value& tgtValue) {
  for (auto j = srcDomain.map.begin(); j != srcDomain.map.end(); ++j) {
    auto name = j->first;
    auto& vd = j->second;
    ValueDomain tgtValDomain;
    switch (vd.type) {
      case ValueDomain::VALUE:
        if (auto newV = translator.valueToTarget(vd.value)) {
          tgtValDomain = ValueDomain::make_value(newV);
        } else {
          unsigned argIndex = translator.getLocalArg(tgtValue, name);
          tgtValDomain = ValueDomain::make_argument(argIndex);
        }
        break;
      case ValueDomain::BUILTIN:
      case ValueDomain::MODULE:
        tgtValDomain = vd;
        break;
      case ValueDomain::ARGUMENT:
        {
          unsigned argIndex = translator.getLocalArg(tgtValue, name);
          tgtValDomain = ValueDomain::make_argument(argIndex);
        }
        break;
    }
    map.insert(std::make_pair(name, tgtValDomain));
  }
}

bool ScopeDomain::mergeFromPrev(ValueTranslator& translator, const ScopeDomain& srcDomain, const mlir::Value& tgtValue) {
  llvm::DenseMap<llvm::StringRef, unsigned> unseen;
  for (auto p : map) {
    unsigned idx = p.second.type == ValueDomain::ARGUMENT ? p.second.argument + 1 : 0;
    unseen.insert(std::make_pair(p.first, idx));
  }
  bool changed = false;
  for (auto i = srcDomain.map.begin(); i != srcDomain.map.end(); ++i) {
    auto name = i->first;
    auto& newDomain = i->second;

    auto p = map.find(name);
    if (p == map.end())
      continue;
    auto& origDomain = p->second;

    unseen.erase(name);

    if (origDomain.type == ValueDomain::ARGUMENT) {
      // Do nothing
    } else if (origDomain == newDomain) {
      // Do nothing
    } else {
      unsigned argIndex = translator.getLocalArg(tgtValue, name);
      p->second = ValueDomain::make_argument(argIndex);
      changed = true;
    }
  }

  for (auto p : unseen) {
    map.erase(p.first);
    if (p.second != 0)
      translator.removeLocalArg(p.second - 1);
  }
  return changed || (unseen.size() > 0);
}

// Create a locals domain from a previous block
void LocalsDomain::populateFromPrev(ValueTranslator& translator, const LocalsDomain& prev) {
  // Iterate through all values in previous block.
  for (auto i = prev.scopeDomains.begin(); i != prev.scopeDomains.end(); ++i) {
    auto val = i->first;
    auto& srcDomain = i->second;
    auto tgtValue = translator.valueToTarget(val);
    if (!tgtValue)
      continue;
    auto p = scopeDomains.insert(std::make_pair(val, ScopeDomain()));
    if (!p.second) {
      fatal_error("Translator maps two values to single value.");
    }
    auto& tgtDomain = p.first->second;
    tgtDomain.initializeFromPrev(translator, srcDomain, tgtValue);
  }
}

/**
 * `x.mergeFromPrev(trans, prev) applies the translator and takes the
 * domain containing facts true in both x and trans(prev).
 */
bool LocalsDomain::mergeFromPrev(ValueTranslator& translator, const LocalsDomain& prev) {
  // Iterate through all values in previous block.
  llvm::DenseSet<mlir::Value> unseen;
  for (auto i = scopeDomains.begin(); i != scopeDomains.end(); ++i) {
    unseen.insert(i->first);
  }

  bool changed = false;
  for (auto i = prev.scopeDomains.begin(); i != prev.scopeDomains.end(); ++i) {
    auto val = i->first;
    auto& srcDomain = i->second;
    auto tgtValue = translator.valueToTarget(val);
    if (!tgtValue)
      continue;
    // Lookup domain for target value in this set.
    auto prevTgtIter = this->scopeDomains.find(tgtValue);
    if (prevTgtIter == this->scopeDomains.end())
      continue;
    if (!unseen.erase(tgtValue)) {
      fatal_error("Duplicate values in scope.");
    }
    if (prevTgtIter->second.mergeFromPrev(translator, srcDomain, tgtValue))
      changed = true;
  }
  for (auto v : unseen) {
    scopeDomains.erase(v);
  }
  return (unseen.size() > 0) || changed;
}
