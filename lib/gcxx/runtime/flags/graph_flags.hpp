// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_FLAGS_GRAPH_FLAGS_HPP_
#define GCXX_RUNTIME_FLAGS_GRAPH_FLAGS_HPP_

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_FLAGS_BEGIN

enum class graphCreate : details_::flag_t {
  None = 0  // as per cuda decumentation they may make new flags in future so
            // for now just set this
};

enum class graphNodeType {
  Kernel       = GCXX_RUNTIME_BACKEND(GraphNodeTypeKernel),
  Memcpy       = GCXX_RUNTIME_BACKEND(GraphNodeTypeMemcpy),
  Memset       = GCXX_RUNTIME_BACKEND(GraphNodeTypeMemset),
  Host         = GCXX_RUNTIME_BACKEND(GraphNodeTypeHost),
  Graph        = GCXX_RUNTIME_BACKEND(GraphNodeTypeGraph),
  Empty        = GCXX_RUNTIME_BACKEND(GraphNodeTypeEmpty),
  EventWait    = GCXX_RUNTIME_BACKEND(GraphNodeTypeWaitEvent),
  EventRecord  = GCXX_RUNTIME_BACKEND(GraphNodeTypeEventRecord),
  ExtSemSignal = GCXX_RUNTIME_BACKEND(GraphNodeTypeExtSemaphoreSignal),
  ExtSemWait   = GCXX_RUNTIME_BACKEND(GraphNodeTypeExtSemaphoreWait),
  Alloc        = GCXX_RUNTIME_BACKEND(GraphNodeTypeMemAlloc),
  Free         = GCXX_RUNTIME_BACKEND(GraphNodeTypeMemFree),
#if GCXX_CUDA_MODE()
  Conditional = GCXX_RUNTIME_BACKEND(GraphNodeTypeConditional),
#endif
};

#if GCXX_CUDA_MODE()
enum class graphConditionalHandle : details_::flag_t {
  None    = 0,
  Default = cudaGraphCondAssignDefault,
};

enum class graphConditionalNode : details_::flag_t {
  If     = GCXX_RUNTIME_BACKEND(GraphCondTypeIf),
  While  = GCXX_RUNTIME_BACKEND(GraphCondTypeWhile),
  Switch = GCXX_RUNTIME_BACKEND(GraphCondTypeSwitch),
};
#endif

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
#if GCXX_CUDA_MODE()
  ConditionalNodeParams =
    GCXX_RUNTIME_BACKEND(GraphDebugDotFlagsConditionalNodeParams),
#endif
};
inline auto operator|(const graphDebugDot& lhs,
                      const graphDebugDot& rhs) -> graphDebugDot {
  return static_cast<graphDebugDot>(static_cast<details_::flag_t>(lhs) |
                                    static_cast<details_::flag_t>(rhs));
}

enum class graphInstantiate : details_::flag_t {
  None         = 0U,
  AutoFree     = GCXX_RUNTIME_BACKEND(GraphInstantiateFlagAutoFreeOnLaunch),
  Upload       = GCXX_RUNTIME_BACKEND(GraphInstantiateFlagUpload),
  DeviceLaunch = GCXX_RUNTIME_BACKEND(GraphInstantiateFlagDeviceLaunch),
  NodePriority = GCXX_RUNTIME_BACKEND(GraphInstantiateFlagUseNodePriority),
};

GCXX_NAMESPACE_MAIN_FLAGS_END

#endif
