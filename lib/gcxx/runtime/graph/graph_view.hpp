#pragma once
#ifndef GCXX_RUNTIME_GRAPH_GRAPH_VIEW_HPP_
#define GCXX_RUNTIME_GRAPH_GRAPH_VIEW_HPP_

#include <cstddef>
#include <string_view>

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

#include <gcxx/runtime/graph/params/graph_host_node_params.hpp>
#include <gcxx/runtime/graph/params/graph_kernel_node_params.hpp>
#include <gcxx/runtime/graph/params/graph_memcpy3d_params.hpp>
#include <gcxx/runtime/graph/params/graph_memset_params.hpp>

#include <gcxx/runtime/flags/graph_flags.hpp>
#include <gcxx/runtime/memory/span/span.hpp>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN
using deviceGraph_t     = GCXX_RUNTIME_BACKEND(Graph_t);
using deviceGraphNode_t = GCXX_RUNTIME_BACKEND(GraphNode_t);
using deviceEvent_t     = GCXX_RUNTIME_BACKEND(Event_t);
using deviceMemcpyKind  = GCXX_RUNTIME_BACKEND(MemcpyKind);

inline constexpr deviceGraph_t INVALID_GRAPH{nullptr};
inline constexpr deviceGraphNode_t INVALID_GRAPH_NODE{nullptr};

// Type aliases for CUDA graph node parameter structures
using deviceKernelNodeParams_t   = GCXX_RUNTIME_BACKEND(KernelNodeParams);
using deviceMemcpy3DParms_t      = GCXX_RUNTIME_BACKEND(Memcpy3DParms);
using deviceMemsetParams_t       = GCXX_RUNTIME_BACKEND(MemsetParams);
using deviceHostNodeParams_t     = GCXX_RUNTIME_BACKEND(HostNodeParams);
using deviceMemAllocNodeParams_t = GCXX_RUNTIME_BACKEND(MemAllocNodeParams);
using deviceGraphEdgeData_t      = GCXX_RUNTIME_BACKEND(GraphEdgeData);
using deviceGraphNodeParams_t    = GCXX_RUNTIME_BACKEND(GraphNodeParams);
using deviceExternalSemaphoreSignalNodeParams_t =
  GCXX_RUNTIME_BACKEND(ExternalSemaphoreSignalNodeParams);
using deviceExternalSemaphoreWaitNodeParams_t =
  GCXX_RUNTIME_BACKEND(ExternalSemaphoreWaitNodeParams);

#if GCXX_CUDA_MODE
using deviceGraphConditionalHandle_t =
  GCXX_RUNTIME_BACKEND(GraphConditionalHandle);
#endif

GCXX_NAMESPACE_MAIN_DETAILS_END


GCXX_NAMESPACE_MAIN_BEGIN
using deviceGraphNode_t = details_::deviceGraphNode_t;

#if GCXX_CUDA_MODE
using deviceGraphConditionalHandle_t = details_::deviceGraphConditionalHandle_t;
#endif

class GraphView {
 protected:
  using deviceGraph_t    = details_::deviceGraph_t;
  using deviceEvent_t    = details_::deviceEvent_t;
  using deviceMemcpyKind = details_::deviceMemcpyKind;

  deviceGraph_t graph_{details_::INVALID_GRAPH};

 public:
  GCXX_FHC GraphView() = default;
  GCXX_FHC GraphView(deviceGraph_t rawgraph);
  GCXX_FHC auto getRawGraph() const -> deviceGraph_t;
  GCXX_FH auto SaveDotfile(std::string_view, flags::graphDebugDot) const
    -> void;
  GCXX_FH auto GetNumNodes() const -> size_t;
  GCXX_FH auto GetNumEdges() const -> size_t;
  GCXX_FH auto Clone() const -> GraphView;

#if GCXX_CUDA_MODE
  GCXX_FH auto CreateConditionalHandle(
    unsigned int defaultLaunchValue,
    flags::graphConditionalHandle flag = flags::graphConditionalHandle::None)
    -> deviceGraphConditionalHandle_t;

  GCXX_FD static auto SetConditional(deviceGraphConditionalHandle_t,
                                     unsigned int) -> void;
#endif
  // ════════════════════════════════════════════════════════════════════════
  // Graph Node Addition Methods
  // ════════════════════════════════════════════════════════════════════════

  GCXX_FH auto addChildGraphNode(
    const GraphView& childGraph,
    const deviceGraphNode_t* pDependencies = nullptr,
    std::size_t numDependencies            = 0) -> deviceGraphNode_t;
  GCXX_FH auto addDependencies(const deviceGraphNode_t* from,
                               const deviceGraphNode_t* to,
                               std::size_t numDependencies) -> void;

  GCXX_FH auto addEmptyNode(const deviceGraphNode_t* pDependencies = nullptr,
                            std::size_t numDependencies            = 0)
    -> deviceGraphNode_t;

  GCXX_FH auto addEventRecordNode(
    const EventView event, const deviceGraphNode_t* pDependencies = nullptr,
    std::size_t numDependencies = 0) -> deviceGraphNode_t;

  GCXX_FH auto addEventWaitNode(
    const EventView event, const deviceGraphNode_t* pDependencies = nullptr,
    std::size_t numDependencies = 0) -> deviceGraphNode_t;

  GCXX_FH auto addHostNode(const deviceHostNodeParams_t* params,
                           const deviceGraphNode_t* pDependencies = nullptr,
                           std::size_t numDependencies            = 0)
    -> deviceGraphNode_t;

  GCXX_FH auto addKernelNode(const deviceKernelNodeParams_t* params,
                             const deviceGraphNode_t* pDependencies = nullptr,
                             std::size_t numDependencies            = 0)
    -> deviceGraphNode_t;

  // GCXX_FH auto addMemAllocNode(,
  //                              const deviceGraphNode_t* pDependencies =
  //                              nullptr, std::size_t numDependencies = 0)
  //   -> deviceGraphNode_t;

  GCXX_FH auto addMemFreeNode(void* dptr,
                              const deviceGraphNode_t* pDependencies = nullptr,
                              std::size_t numDependencies            = 0)
    -> deviceGraphNode_t;

  GCXX_FH auto addMemcpyNode(const deviceMemcpy3DParams_t* params,
                             const deviceGraphNode_t* pDependencies = nullptr,
                             std::size_t numDependencies            = 0)
    -> deviceGraphNode_t;

  GCXX_FH auto addMemcpyNode1D(void* dst, const void* src,
                               std::size_t countBytes,
                               const deviceGraphNode_t* pDependencies = nullptr,
                               std::size_t numDependencies            = 0)
    -> deviceGraphNode_t;

  // GCXX_FH auto addMemcpyNodeFromSymbol(
  //   , const deviceGraphNode_t* pDependencies = nullptr,
  //   std::size_t numDependencies        = 0) -> deviceGraphNode_t;
  // GCXX_FH auto addcudaGraphAddMemcpyNodeToSymbol(
  //   , const deviceGraphNode_t* pDependencies = nullptr,
  //   std::size_t numDependencies        = 0) -> deviceGraphNode_t;

  GCXX_FH auto addMemsetNode(const deviceMemsetParams_t* params,
                             const deviceGraphNode_t* pDependencies = nullptr,
                             std::size_t numDependencies            = 0)
    -> deviceGraphNode_t;


  // // general version to add anything will make a staic dispatch with diffrent
  // // parameters
  // GCXX_FH auto addNode() -> deviceGraphNode_t;
};

GCXX_NAMESPACE_MAIN_END

#include <gcxx/runtime/details/graph/graph_view.inl>

#include <gcxx/macros/undefine_macros.hpp>
#endif
