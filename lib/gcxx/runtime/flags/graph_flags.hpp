#pragma once
#ifndef GCXX_RUNTIME_FLAGS_GRAPH_FLAGS_HPP_
#define GCXX_RUNTIME_FLAGS_GRAPH_FLAGS_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

GCXX_NAMESPACE_MAIN_FLAGS_BEGIN

enum class graphCreate : details_::flag_t {
  None = 0  // as per cuda decumentation they may make new flags in future so
            // for now just set this
};

enum class graphConditionalHandle : details_::flag_t {
  None    = 0,
  Default = cudaGraphCondAssignDefault,
};

enum class graphDebugDot : details_::flag_t {
  Verbose          = GCXX_RUNTIME_BACKEND(GraphDebugDotFlagsVerbose),
  KernelNodeParams = GCXX_RUNTIME_BACKEND(GraphDebugDotFlagsKernelNodeParams),
  MemcpyNodeParams = GCXX_RUNTIME_BACKEND(GraphDebugDotFlagsMemcpyNodeParams),
  MemsetNodeParams = GCXX_RUNTIME_BACKEND(GraphDebugDotFlagsMemsetNodeParams),
  HostNodeParams   = GCXX_RUNTIME_BACKEND(GraphDebugDotFlagsHostNodeParams),
  EventNodeParams  = GCXX_RUNTIME_BACKEND(GraphDebugDotFlagsEventNodeParams),
  ExtSemasSignalNodeParams =
    GCXX_RUNTIME_BACKEND(GraphDebugDotFlagsExtSemasSignalNodeParams),
  ExtSemasWaitNodeParams =
    GCXX_RUNTIME_BACKEND(GraphDebugDotFlagsExtSemasWaitNodeParams),
  KernelNodeAttributes =
    GCXX_RUNTIME_BACKEND(GraphDebugDotFlagsKernelNodeAttributes),
  Handles = GCXX_RUNTIME_BACKEND(GraphDebugDotFlagsHandles),
#if GCXX_CUDA_MODE
  ConditionalNodeParams =
    GCXX_RUNTIME_BACKEND(GraphDebugDotFlagsConditionalNodeParams),
#endif
};

inline auto operator|(const graphDebugDot& lhs, const graphDebugDot& rhs)
  -> graphDebugDot {
  return static_cast<graphDebugDot>(static_cast<details_::flag_t>(lhs) |
                                    static_cast<details_::flag_t>(rhs));
}

GCXX_NAMESPACE_MAIN_FLAGS_END

#endif
