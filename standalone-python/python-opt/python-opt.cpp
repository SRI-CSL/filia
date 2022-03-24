#include <stdio.h>
#include <mlir/Parser.h>
#include <mlir/InitAllDialects.h>
#include <llvm/ADT/ImmutableMap.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/SmallSet.h>

#include "Python/PythonDialect.h"
#include "Python/PythonOps.h"

mlir::Block::OpListType& moduleOps(mlir::Operation& modulePtr) {
  if (modulePtr.getName().getIdentifier().str() != "builtin.module") {
    fprintf(stderr, "Expected module (found %s)", modulePtr.getName().getIdentifier().str().c_str());
    exit(-1);
  }

  if (modulePtr.getNumRegions() != 1) {
    fprintf(stderr, "Expected single region in module.\n");
    exit(-1);
  }

  auto& blks = modulePtr.getRegion(0).getBlocks();
  if (blks.size() != 1) {
    fprintf(stderr, "Expected single root block.\n");
    exit(-1);
  }
  return blks.begin()->getOperations();
}

static
void printAttrs(mlir::Operation& op) {
  for(auto& attr : op.getAttrs()) {
    llvm::outs() << "Attr: " << attr.getName() << "\n";
  }
}

class StringWrapper {
  llvm::StringRef x;
public:
  explicit StringWrapper(const llvm::StringRef& x) : x(x) {}

  const llvm::StringRef& get() const { return x; }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddString(x);
  }
};

bool operator==(const StringWrapper& x, const StringWrapper& y) {
  return x.get() == y.get();
}

bool operator<(const StringWrapper& x, const StringWrapper& y) {
  return x.get() < y.get();
}

using LocalsDomainFactory = llvm::ImutAVLFactory<llvm::ImutKeyValueInfo<StringWrapper, mlir::detail::ValueImpl*>>;


static
llvm::SmallSet<llvm::StringRef, 4> getBuiltins() {
  static llvm::SmallSet<llvm::StringRef, 4> r;
  r.insert("eval");
  r.insert("getattr");
  r.insert("open");
  r.insert("print");
  return r;
}

bool isBuiltin(const llvm::StringRef& name) {
  static auto builtins(getBuiltins());
  return builtins.contains(name);
}

class LocalsDomain {
private:
  using ValInfo = llvm::ImutKeyValueInfo<StringWrapper, mlir::detail::ValueImpl*>;
  using TreeTy = llvm::ImutAVLTree<ValInfo>;

  LocalsDomainFactory *Factory;
  TreeTy* Root; // FIXME.  This currently has a memory leak because we do not free Root.

public:
  LocalsDomain() : Factory(nullptr), Root(nullptr) {}

  explicit LocalsDomain(LocalsDomainFactory *f)
    : Factory(f), Root(nullptr) {
  }

  LocalsDomain(const LocalsDomain&) = default;

  LocalsDomain(const LocalsDomain& d, const llvm::StringRef& name, mlir::Value value)
    : Factory(d.Factory), Root(d.Factory->add(d.Root, std::make_pair(StringWrapper(name), value.getImpl()))) {
  }

  operator bool() const {
    return Factory != nullptr;
  }

  mlir::Value getValue(const llvm::StringRef& name) {
    if (Root) {
      TreeTy* T = Root->find(StringWrapper(name));
      if (T)
        return T->getValue().second;
    }
    return mlir::Value();
  }
};

template<typename K, typename V>
using HTType = llvm::ScopedHashTable<K, V, llvm::DenseMapInfo<K>, llvm::MallocAllocator>;

//using ValueLocalsAllocator = LoadMapAllocator<mlir::Value, LocalsDomain>;
using ValueLocalsMap = HTType<mlir::Value, LocalsDomain>;
using ValueLocalsMapScope = llvm::ScopedHashTableScope<mlir::Value, LocalsDomain>;

mlir::FuncOp visitFunction(mlir::Operation& fun) {
  auto location = fun.getLoc();

  auto symNameAttr = fun.getAttrOfType<mlir::StringAttr>("sym_name");
  if (!symNameAttr) {
    fprintf(stderr, "Could not find function name.\n");
    exit(-1);
  }


  auto funTypeAttr = fun.getAttrOfType<mlir::TypeAttr>("type");
  if (!funTypeAttr) {
    fprintf(stderr, "Could not find function type.\n");
    exit(-1);
  }
  auto funType = funTypeAttr.getValue().dyn_cast<mlir::FunctionType>();


//    // This is a generic function, the return type will be inferred later.
    // Arguments type are uniformly unranked tensors.
 //   llvm::SmallVector<mlir::Type, 4> argTypes(proto.getArgs().size(),
//                                              getType(VarType{}));
//    auto funcType = builder.getFunctionType(argTypes, llvm::None);

  mlir::FuncOp result(mlir::FuncOp::create(location, symNameAttr, funType));
  if (fun.getNumRegions() != 1) {
    fprintf(stderr, "Expected single region in function.\n");
    exit(-1);
  }

  auto& blks = fun.getRegion(0).getBlocks();

  if (blks.size() == 0) {
    return result;
  }

  // Let's start the body of the function now!
  // In MLIR the entry block of the function is special: it must have the same
  // argument list as the function itself.
  mlir::Block* newBlockPtr = result.addEntryBlock();

  auto blockPtr = blks.begin();
  mlir::BlockAndValueMapping mapper;

  ValueLocalsMap localsMap;
  ValueLocalsMapScope localsScope(localsMap);

  // Used to create locals domains
  LocalsDomainFactory localsFactory;

  while (true) {
    mlir::OpBuilder builder(mlir::OpBuilder::atBlockBegin(newBlockPtr));

    for (auto opPtr = blockPtr->begin(); opPtr != blockPtr->end(); ++opPtr) {
      auto opName = opPtr->getName().getIdentifier().str();
      if (opName == "python.empty_locals") {
        assert (opPtr->getNumResults() == 1);
        auto r = opPtr->getResult(0);
        localsMap.insert(r, LocalsDomain(&localsFactory));

        builder.clone(*opPtr, mapper);
      } else if (opName == "python.local_set") {
        assert(opPtr->getNumOperands() == 2);
        auto m = localsMap.lookup(opPtr->getOperand(0));
        assert(m);

        auto v = opPtr->getOperand(1);
        auto newVal = mapper.lookupOrNull(v);
        assert(newVal);

        auto nameAttr = opPtr->getAttrOfType<mlir::StringAttr>("name");
        assert(nameAttr);

        assert (opPtr->getNumResults() == 1);
        auto r = opPtr->getResult(0);
        localsMap.insert(r, LocalsDomain(m, nameAttr.getValue(), newVal));

        builder.clone(*opPtr, mapper);
      } else if (opName == "python.get") {
        assert(opPtr->getNumOperands() == 1);
        auto m = localsMap.lookup(opPtr->getOperand(0));
        assert(m);
        auto nameAttr = opPtr->getAttrOfType<mlir::StringAttr>("name");
        assert(nameAttr);
        auto name = nameAttr.getValue();

        assert (opPtr->getNumResults() == 1);
        auto origResult = opPtr->getResult(0);

        if (auto newVal = m.getValue(name)) {
          mapper.map(origResult, newVal);
        } else if (auto builtin = mlir::python::symbolizeBuiltinAttr(name)) {
          auto location = opPtr->getLoc();
          auto b = builder.create<mlir::python::Builtin>(location, mlir::python::stringifyEnum(builtin.getValue()));
          mapper.map(origResult, b.result());
//          builder.clone(*opPtr, mapper);
        } else {
          builder.clone(*opPtr, mapper);
        }
      } else {
        builder.clone(*opPtr, mapper);
      }
    }
    ++blockPtr;
    if (blockPtr == blks.end())
      break;
    newBlockPtr = result.addBlock();
  }
  return result;
}

void initContext(mlir::MLIRContext& context, const mlir::DialectRegistry& registry) {
  context.appendDialectRegistry(registry);
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::python::PythonDialect>();
}

int main(int argc, const char** argv) {
  if (argc != 2) {
    fprintf(stderr, "Please provide the path to read.\n");
    return -1;
  }
  const char* path = argv[1];

  mlir::MLIRContext inputContext;

//  context.allowUnregisteredDialects();
  mlir::DialectRegistry registry;
  registry.insert<mlir::python::PythonDialect>();
  initContext(inputContext, registry);
  mlir::Block block;
  mlir::LogicalResult r = mlir::parseSourceFile(path, &block, &inputContext);
  if (r.failed()) {
    fprintf(stderr, "Failed to parse mlir file file.\n");
    return -1;
  }
  if (block.getOperations().size() != 1) {
    fprintf(stderr, "Missing module.\n");
    return -1;
  }

  mlir::MLIRContext resolvedContext;
  initContext(resolvedContext, registry);

   mlir::OpBuilder builder(&resolvedContext);
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
  auto theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

  auto& ops = moduleOps(*block.begin());
  for (auto moduleOpPtr = ops.begin(); moduleOpPtr != ops.end(); ++moduleOpPtr) {
    auto opName = moduleOpPtr->getName().getIdentifier().str();
    if (opName == "builtin.func") {
      auto func = visitFunction(*moduleOpPtr);
      if (func)
        theModule.push_back(func);
    } else {
      llvm::outs() << "Unknown Operator " << opName << "\n";
    }
  }

  theModule.dump();

  return 0;
}