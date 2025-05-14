#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"

using namespace mlir;

namespace {

struct TestMembarPass
    : public PassWrapper<TestMembarPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMembarPass);

  StringRef getArgument() const final { return "test-print-membar"; }
  StringRef getDescription() const final {
    return "print the result of the allocation pass";
  }

  void runOnOperation() override {
    Operation *operation = getOperation();
    ModuleOp moduleOp = cast<ModuleOp>(operation);
    // Print all ops after membar pass
    ModuleAllocation allocation(moduleOp);
    ModuleMembarAnalysis membarPass(&allocation,
                                    [](void *a, void *b) { return false; });
    membarPass.run();
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestMembarPass() { PassRegistration<TestMembarPass>(); }
} // namespace test
} // namespace mlir
