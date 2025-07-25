#include "PartitionBuilder.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
// #include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/WarpSpecialization.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

//===----------------------------------------------------------------------===//
// rewritePartitionDependenies
//===----------------------------------------------------------------------===//

LogicalResult triton::gpu::rewritePartitionDependencies(scf::ForOp &loop) {
  return failure();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUREWRITEPARTITIONDEPENDENCIES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct RewritePartitionDependencies
    : triton::gpu::impl::TritonGPURewritePartitionDependenciesBase<
          RewritePartitionDependencies> {
  using TritonGPURewritePartitionDependenciesBase::
      TritonGPURewritePartitionDependenciesBase;

  void runOnOperation() override;
};
} // namespace

void RewritePartitionDependencies::runOnOperation() {
  // Collect for loops to warp specialize. This pass expects the loop to
  // already be scheduled.
  SmallVector<scf::ForOp> loops;
  getOperation().walk([&](scf::ForOp loop) {
    if (loop->hasAttrOfType<ArrayAttr>(kPartitionStagesAttrName))
      loops.push_back(loop);
  });

  for (scf::ForOp loop : loops) {
    if (failed(rewritePartitionDependencies(loop)))
      return signalPassFailure();
  }
}
