// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_HANDLES_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_HANDLES_HPP_

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN

// ╔════════════════════════════════════════════════════════╗
// ║                     Error Handles                      ║
// ╚════════════════════════════════════════════════════════╝
using deviceError_t                     = GCXX_RUNTIME_BACKEND(Error_t);
GCXX_CXPR inline auto deviceErrSuccess  = GCXX_RUNTIME_BACKEND(Success);
GCXX_CXPR inline auto deviceErrNotReady = GCXX_RUNTIME_BACKEND(ErrorNotReady);

// ╔════════════════════════════════════════════════════════╗
// ║                     Stream Handles                      ║
// ╚════════════════════════════════════════════════════════╝
using deviceStream_t = GCXX_RUNTIME_BACKEND(Stream_t);
inline static constexpr deviceStream_t NULL_STREAM{nullptr};  // NOLINT

// clang-format off
inline static const auto INVALID_STREAM = reinterpret_cast<deviceStream_t>(~0ULL);  // NOLINT
// clang-format on

// ╔════════════════════════════════════════════════════════╗
// ║                     Event Handles                      ║
// ╚════════════════════════════════════════════════════════╝
using deviceEvent_t = GCXX_RUNTIME_BACKEND(Event_t);
inline static constexpr deviceEvent_t INVALID_EVENT{nullptr};  // NOLINT


// ╔════════════════════════════════════════════════════════╗
// ║                     Graph Handles                      ║
// ╚════════════════════════════════════════════════════════╝
using deviceGraph_t           = GCXX_RUNTIME_BACKEND(Graph_t);
using deviceGraphNode_t       = GCXX_RUNTIME_BACKEND(GraphNode_t);
using deviceGraphExec_t       = GCXX_RUNTIME_BACKEND(GraphExec_t);
using deviceGraphNodeParams_t = GCXX_RUNTIME_BACKEND(GraphNodeParams);
using deviceGraphNodeType_t   = GCXX_RUNTIME_BACKEND(GraphNodeType);

using deviceHostCallBackFn_t   = GCXX_RUNTIME_BACKEND(HostFn_t);
using deviceHostNodeParams_t   = GCXX_RUNTIME_BACKEND(HostNodeParams);
using deviceKernelNodeParams_t = GCXX_RUNTIME_BACKEND(KernelNodeParams);
using deviceMemcpy3DParams_t   = GCXX_RUNTIME_BACKEND(Memcpy3DParms);
using deviceMemsetParams_t     = GCXX_RUNTIME_BACKEND(MemsetParams);

inline constexpr deviceGraph_t INVALID_GRAPH{nullptr};           // NOLINT
inline constexpr deviceGraphExec_t INVALID_GRAPH_EXEC{nullptr};  // NOLINT
inline constexpr deviceGraphNode_t INVALID_GRAPH_NODE{nullptr};  // NOLINT

#if GCXX_CUDA_MODE
using deviceGraphConditionalHandle_t =
  GCXX_RUNTIME_BACKEND(GraphConditionalHandle);
#endif


GCXX_NAMESPACE_MAIN_DRIVER_END

#endif