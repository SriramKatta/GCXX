// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_HANDLES_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_HANDLES_HPP_

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN()

// ╔════════════════════════════════════════════════════════╗
// ║                     Device Handles                     ║
// ╚════════════════════════════════════════════════════════╝
using deviceProp_t         = GCXX_RUNTIME_BACKEND_ALT(DeviceProp, DeviceProp_t);
using deviceAttribute_t    = GCXX_RUNTIME_BACKEND_ALT(DeviceAttr,
                                                      DeviceAttribute_t);
using funcCacheConfig_t    = GCXX_RUNTIME_BACKEND_ALT(FuncCache, FuncCache_t);
using deviceLimit_t        = GCXX_RUNTIME_BACKEND_ALT(Limit, Limit_t);
using deviceMemPool_t      = GCXX_RUNTIME_BACKEND(MemPool_t);
using deviceMemPoolProps_t = GCXX_RUNTIME_BACKEND(MemPoolProps);
using deviceMemPoolPtrExportData_t = GCXX_RUNTIME_BACKEND(MemPoolPtrExportData);
using deviceMemPoolAttr_t          = GCXX_RUNTIME_BACKEND(MemPoolAttr);
using deviceMemLocation_t          = GCXX_RUNTIME_BACKEND(MemLocation);
using deviceMemAllocationType_t    = GCXX_RUNTIME_BACKEND(MemAllocationType);
using deviceMemAllocationHandleType_t =
  GCXX_RUNTIME_BACKEND(MemAllocationHandleType);
using deviceMemAccessFlags_t = GCXX_RUNTIME_BACKEND(MemAccessFlags);
using deviceMemAccessDesc_t  = GCXX_RUNTIME_BACKEND(MemAccessDesc);
using deviceP2PAttr_t        = GCXX_RUNTIME_BACKEND(DeviceP2PAttr);
using deviceIpcEventHandle_t = GCXX_RUNTIME_BACKEND(IpcEventHandle_t);
using deviceIpcMemHandle_t   = GCXX_RUNTIME_BACKEND(IpcMemHandle_t);

// ╔════════════════════════════════════════════════════════╗
// ║                   Execution Handles                    ║
// ╚════════════════════════════════════════════════════════╝
using deviceLaunchConfig_t = GCXX_RUNTIME_BACKEND(LaunchConfig_t);
using deviceKenel_t        = GCXX_RUNTIME_BACKEND(Kernel_t);

// ╔════════════════════════════════════════════════════════╗
// ║                     Error Handles                      ║
// ╚════════════════════════════════════════════════════════╝
using deviceError_t                     = GCXX_RUNTIME_BACKEND(Error_t);
GCXX_CXPR inline auto deviceErrSuccess  = GCXX_RUNTIME_BACKEND(Success);
GCXX_CXPR inline auto deviceErrNotReady = GCXX_RUNTIME_BACKEND(ErrorNotReady);

// ╔════════════════════════════════════════════════════════╗
// ║                     Stream Handles                      ║
// ╚════════════════════════════════════════════════════════╝
using deviceStream_t              = GCXX_RUNTIME_BACKEND(Stream_t);
using deviceStreamCallback_t      = GCXX_RUNTIME_BACKEND(StreamCallback_t);
using deviceStreamCaptureMode_t   = GCXX_RUNTIME_BACKEND(StreamCaptureMode);
using deviceStreamCaptureStatus_t = GCXX_RUNTIME_BACKEND(StreamCaptureStatus);
inline static constexpr deviceStream_t NULL_STREAM{nullptr};  // NOLINT

// clang-format off
inline static const auto INVALID_STREAM = reinterpret_cast<deviceStream_t>(~0ULL);  // NOLINT
// clang-format on

#if GCXX_CUDA_MODE()
using deviceStreamAttrID_t    = GCXX_RUNTIME_BACKEND(StreamAttrID);
using deviceStreamAttrValue_t = GCXX_RUNTIME_BACKEND(StreamAttrValue);
#endif

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
using deviceGraphEdgeData_t   = GCXX_RUNTIME_BACKEND(GraphEdgeData);

using deviceHostCallBackFn_t   = GCXX_RUNTIME_BACKEND(HostFn_t);
using deviceHostNodeParams_t   = GCXX_RUNTIME_BACKEND(HostNodeParams);
using deviceKernelNodeParams_t = GCXX_RUNTIME_BACKEND(KernelNodeParams);
using deviceMemcpy3DParams_t   = GCXX_RUNTIME_BACKEND(Memcpy3DParms);
using deviceMemsetParams_t     = GCXX_RUNTIME_BACKEND(MemsetParams);
using deviceMemcpyKind_t       = GCXX_RUNTIME_BACKEND(MemcpyKind);

#if GCXX_CUDA_MODE()
using deviceExternalSemaphoreSignalNodeParams_t =
  GCXX_RUNTIME_BACKEND(ExternalSemaphoreSignalNodeParams);
using deviceExternalSemaphoreWaitNodeParams_t =
  GCXX_RUNTIME_BACKEND(ExternalSemaphoreWaitNodeParams);
using deviceGraphExecUpdateResultInfo_t =
  GCXX_RUNTIME_BACKEND(GraphExecUpdateResultInfo);
using deviceGraphInstantiateParams_t =
  GCXX_RUNTIME_BACKEND(GraphInstantiateParams);
using deviceGraphMemAttributeType_t =
  GCXX_RUNTIME_BACKEND(GraphMemAttributeType);
using deviceKernelNodeAttrID_t    = GCXX_RUNTIME_BACKEND(KernelNodeAttrID);
using deviceKernelNodeAttrValue_t = GCXX_RUNTIME_BACKEND(KernelNodeAttrValue);
using deviceMemAllocNodeParams_t  = GCXX_RUNTIME_BACKEND(MemAllocNodeParams);
using deviceUserObject_t          = GCXX_RUNTIME_BACKEND(UserObject_t);
using deviceGraphDeviceNode_t     = GCXX_RUNTIME_BACKEND(GraphDeviceNode_t);
using deviceGraphKernelNodeUpdate_t =
  GCXX_RUNTIME_BACKEND(GraphKernelNodeUpdate);
#endif

inline constexpr deviceGraph_t INVALID_GRAPH{nullptr};           // NOLINT
inline constexpr deviceGraphExec_t INVALID_GRAPH_EXEC{nullptr};  // NOLINT
inline constexpr deviceGraphNode_t INVALID_GRAPH_NODE{nullptr};  // NOLINT

#if GCXX_CUDA_MODE()
using deviceGraphConditionalHandle_t =
  GCXX_RUNTIME_BACKEND(GraphConditionalHandle);
#endif


GCXX_NAMESPACE_MAIN_DRIVER_END()

#endif
