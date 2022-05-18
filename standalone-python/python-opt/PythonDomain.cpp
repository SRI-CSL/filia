#include "PythonDomain.h"
#include "ValueTranslator.h"

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
