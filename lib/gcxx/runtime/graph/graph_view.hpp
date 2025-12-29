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

  GCXX_FH auto AddChildGraphNode(
    const GraphView& childGraph,
    const deviceGraphNode_t* pDependencies = nullptr,
    std::size_t numDependencies            = 0) -> deviceGraphNode_t;
  GCXX_FH auto AddDependencies(const deviceGraphNode_t* from,
                               const deviceGraphNode_t* to,
                               std::size_t numDependencies) -> void;

  GCXX_FH auto AddEmptyNode(const deviceGraphNode_t* pDependencies = nullptr,
                            std::size_t numDependencies            = 0)
    -> deviceGraphNode_t;

  GCXX_FH auto AddEventRecordNode(
    const EventView event, const deviceGraphNode_t* pDependencies = nullptr,
    std::size_t numDependencies = 0) -> deviceGraphNode_t;

  GCXX_FH auto AddEventWaitNode(
    const EventView event, const deviceGraphNode_t* pDependencies = nullptr,
    std::size_t numDependencies = 0) -> deviceGraphNode_t;

  GCXX_FH auto AddHostNode(const deviceHostNodeParams_t* params,
                           const deviceGraphNode_t* pDependencies = nullptr,
                           std::size_t numDependencies            = 0)
    -> deviceGraphNode_t;

  GCXX_FH auto AddKernelNode(const deviceKernelNodeParams_t* params,
                             const deviceGraphNode_t* pDependencies = nullptr,
                             std::size_t numDependencies            = 0)
    -> deviceGraphNode_t;

  // GCXX_FH auto AddMemAllocNode(,
  //                              const deviceGraphNode_t* pDependencies =
  //                              nullptr, std::size_t numDependencies = 0)
  //   -> deviceGraphNode_t;

  GCXX_FH auto AddMemFreeNode(void* dptr,
                              const deviceGraphNode_t* pDependencies = nullptr,
                              std::size_t numDependencies            = 0)
    -> deviceGraphNode_t;

  GCXX_FH auto AddMemcpyNode(const deviceMemcpy3DParams_t* params,
                             const deviceGraphNode_t* pDependencies = nullptr,
                             std::size_t numDependencies            = 0)
    -> deviceGraphNode_t;

  GCXX_FH auto AddMemcpyNode1D(void* dst, const void* src,
                               std::size_t countBytes,
                               const deviceGraphNode_t* pDependencies = nullptr,
                               std::size_t numDependencies            = 0)
    -> deviceGraphNode_t;

  // GCXX_FH auto AddMemcpyNodeFromSymbol(
  //   , const deviceGraphNode_t* pDependencies = nullptr,
  //   std::size_t numDependencies        = 0) -> deviceGraphNode_t;
  // GCXX_FH auto AddcudaGraphAddMemcpyNodeToSymbol(
  //   , const deviceGraphNode_t* pDependencies = nullptr,
  //   std::size_t numDependencies        = 0) -> deviceGraphNode_t;

  GCXX_FH auto AddMemsetNode(const deviceMemsetParams_t* params,
                             const deviceGraphNode_t* pDependencies = nullptr,
                             std::size_t numDependencies            = 0)
    -> deviceGraphNode_t;


  // // general version to Add anything will make a staic dispatch with diffrent
  // // parameters
  // GCXX_FH auto AddNode() -> deviceGraphNode_t;


  // ════════════════════════════════════════════════════════════════════════
  // CPP style Graph Node Addition Methods
  // ════════════════════════════════════════════════════════════════════════

  GCXX_FH auto AddChildGraphNode(
    const GraphView& childGraph,
    gcxx::span<const deviceGraphNode_t> pDependencies = {})
    -> deviceGraphNode_t;


  GCXX_FH auto AddDependencies(gcxx::span<const deviceGraphNode_t> from,
                               gcxx::span<const deviceGraphNode_t> to) -> void;

  GCXX_FH auto AddEmptyNode(
    gcxx::span<const deviceGraphNode_t> pDependencies = {})
    -> deviceGraphNode_t;

  GCXX_FH auto AddEventRecordNode(
    const EventView event,
    gcxx::span<const deviceGraphNode_t> pDependencies = {})
    -> deviceGraphNode_t;

  GCXX_FH auto AddEventWaitNode(
    const EventView event,
    gcxx::span<const deviceGraphNode_t> pDependencies = {})
    -> deviceGraphNode_t;

  GCXX_FH auto AddHostNode(
    const HostNodeParamsView params,
    gcxx::span<const deviceGraphNode_t> pDependencies = {})
    -> deviceGraphNode_t;

  GCXX_FH auto AddKernelNode(
    const KernelNodeParamsView params,
    gcxx::span<const deviceGraphNode_t> pDependencies = {})
    -> deviceGraphNode_t;

  GCXX_FH auto AddMemFreeNode(
    void* dptr, gcxx::span<const deviceGraphNode_t> pDependencies = {})
    -> deviceGraphNode_t;

  GCXX_FH auto AddMemcpyNode(
    const Memcpy3DParamsView params,
    gcxx::span<const deviceGraphNode_t> pDependencies = {})
    -> deviceGraphNode_t;

  GCXX_FH auto AddMemcpyNode1D(
    void* dst, const void* src, std::size_t countBytes,
    gcxx::span<const deviceGraphNode_t> pDependencies = {})
    -> deviceGraphNode_t;

  GCXX_FH auto AddMemsetNode(
    const MemsetParamsView params,
    gcxx::span<const deviceGraphNode_t> pDependencies = {})
    -> deviceGraphNode_t;
};

GCXX_NAMESPACE_MAIN_END

#include <gcxx/runtime/details/graph/graph_view.inl>

#include <gcxx/macros/undefine_macros.hpp>
#endif
