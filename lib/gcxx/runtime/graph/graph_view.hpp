// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_GRAPH_VIEW_HPP_
#define GCXX_RUNTIME_GRAPH_GRAPH_VIEW_HPP_

#include <cstddef>
#include <string_view>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/flags/graph_flags.hpp>
#include <gcxx/runtime/graph/graph_params.hpp>
#include <gcxx/runtime/memory/spans/spans.hpp>
#include <gcxx/runtime_backend/backend_graph.hpp>

#include <gcxx/runtime/graph/graph_nodes.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


class GraphView;
/// Result struct for AddIfNode with named fields
struct IfNodeResult;
struct IfElseNodeResult;
struct WhileNodeResult;
struct SwitchNodeResult;

class GraphView {
 public:
  using deviceGraph_t           = driver::deviceGraph_t;
  using deviceGraphNode_t       = driver::deviceGraphNode_t;
  using deviceGraphNodeParams_t = driver::deviceGraphNodeParams_t;
#if GCXX_CUDA_MODE()
  using deviceGraphConditionalHandle_t = driver::deviceGraphConditionalHandle_t;
#endif

 protected:
  deviceGraph_t graph_{driver::INVALID_GRAPH};  // NOLINT

 public:
  GCXX_FHC GraphView() = default;
  GCXX_FHC GraphView(deviceGraph_t rawgraph);
  GCXX_FHC auto getRawGraph() const -> deviceGraph_t;
  GCXX_FH
  auto SaveDotfile(std::string_view, flags::graphDebugDot) const -> void;
  GCXX_FH auto GetNumNodes() const -> size_t;
  GCXX_FH auto GetNumEdges() const -> size_t;
  GCXX_FH auto Clone() const -> GraphView;

#if GCXX_CUDA_MODE()
  GCXX_FH
  auto CreateConditionalHandle(
    unsigned int defaultLaunchValue,
    flags::graphConditionalHandle flag = flags::graphConditionalHandle::None)
    -> deviceGraphConditionalHandle_t;

  GCXX_FD
  static auto SetConditional(deviceGraphConditionalHandle_t,
                             unsigned int) -> void;

  GCXX_FH
  auto AddIfNode(deviceGraphConditionalHandle_t condHand,
                 const deviceGraphNode_t* pDependencies = nullptr,
                 std::size_t numDependencies            = 0) -> IfNodeResult;

  GCXX_FH
  auto AddIfElseNode(deviceGraphConditionalHandle_t condHand,
                     const deviceGraphNode_t* pDependencies = nullptr,
                     std::size_t numDependencies = 0) -> IfElseNodeResult;

  GCXX_FH
  auto AddWhileNode(deviceGraphConditionalHandle_t condHand,
                    const deviceGraphNode_t* pDependencies = nullptr,
                    std::size_t numDependencies = 0) -> WhileNodeResult;

  GCXX_FH
  auto AddSwitchNode(deviceGraphConditionalHandle_t condHand,
                     std::size_t numCases,
                     const deviceGraphNode_t* pDependencies = nullptr,
                     std::size_t numDependencies = 0) -> SwitchNodeResult;
#endif
  // ════════════════════════════════════════════════════════════════════════
  // Graph Node Addition Methods
  // ════════════════════════════════════════════════════════════════════════

  GCXX_FH
  auto AddChildGraphNode(const GraphView& childGraph,
                         const deviceGraphNode_t* pDependencies = nullptr,
                         std::size_t numDependencies = 0) -> ChildGraphNodeView;
  GCXX_FH
  auto AddDependencies(const deviceGraphNode_t* from,
                       const deviceGraphNode_t* to,
                       std::size_t numDependencies) -> void;

  GCXX_FH
  auto AddEmptyNode(const deviceGraphNode_t* pDependencies = nullptr,
                    std::size_t numDependencies = 0) -> deviceGraphNode_t;

  GCXX_FH
  auto AddEventRecordNode(const EventView event,
                          const deviceGraphNode_t* pDependencies = nullptr,
                          std::size_t numDependencies = 0) -> deviceGraphNode_t;

  GCXX_FH
  auto AddEventWaitNode(const EventView event,
                        const deviceGraphNode_t* pDependencies = nullptr,
                        std::size_t numDependencies = 0) -> deviceGraphNode_t;

  GCXX_FH
  auto AddHostNode(const HostNodeParamsView::deviceHostNodeParams_t* params,
                   const deviceGraphNode_t* pDependencies = nullptr,
                   std::size_t numDependencies = 0) -> deviceGraphNode_t;

  GCXX_FH
  auto AddKernelNode(
    const KernelNodeParamsView::deviceKernelNodeParams_t* params,
    const deviceGraphNode_t* pDependencies = nullptr,
    std::size_t numDependencies            = 0) -> deviceGraphNode_t;

  // GCXX_FHauto AddMemAllocNode(,
  //                              const deviceGraphNode_t* pDependencies =
  //                              nullptr, std::size_t numDependencies = 0)
  //   -> deviceGraphNode_t;

  GCXX_FH
  auto AddMemFreeNode(void* dptr,
                      const deviceGraphNode_t* pDependencies = nullptr,
                      std::size_t numDependencies = 0) -> deviceGraphNode_t;

  GCXX_FH
  auto AddMemcpyNode(const Memcpy3DParamsView::deviceMemcpy3DParams_t* params,
                     const deviceGraphNode_t* pDependencies = nullptr,
                     std::size_t numDependencies = 0) -> deviceGraphNode_t;

  GCXX_FH
  auto AddMemcpyNode1D(void* dst, const void* src, std::size_t countBytes,
                       const deviceGraphNode_t* pDependencies = nullptr,
                       std::size_t numDependencies = 0) -> deviceGraphNode_t;

  // GCXX_FHauto AddMemcpyNodeFromSymbol(
  //   , const deviceGraphNode_t* pDependencies = nullptr,
  //   std::size_t numDependencies        = 0) -> deviceGraphNode_t;
  // GCXX_FHauto AddcudaGraphAddMemcpyNodeToSymbol(
  //   , const deviceGraphNode_t* pDependencies = nullptr,
  //   std::size_t numDependencies        = 0) -> deviceGraphNode_t;

  GCXX_FH
  auto AddMemsetNode(const MemsetParamsView::deviceMemsetParams_t* params,
                     const deviceGraphNode_t* pDependencies = nullptr,
                     std::size_t numDependencies = 0) -> deviceGraphNode_t;


  // // general version to Add anything will make a staic dispatch with diffrent
  // // parameters
  // GCXX_FHauto AddNode() -> deviceGraphNode_t;

  // ════════════════════════════════════════════════════════════════════════
  // CPP style Graph Node Addition Methods
  // ════════════════════════════════════════════════════════════════════════

  GCXX_FH
  auto AddChildGraphNode(const GraphView& childGraph,
                         gcxx::span<const deviceGraphNode_t> pDependencies = {})
    -> ChildGraphNodeView;


  GCXX_FH
  auto AddDependencies(gcxx::span<const deviceGraphNode_t> from,
                       gcxx::span<const deviceGraphNode_t> to) -> void;

  GCXX_FH
  auto AddEmptyNode(gcxx::span<const deviceGraphNode_t> pDependencies = {})
    -> deviceGraphNode_t;

  GCXX_FH
  auto AddEventRecordNode(const EventView event,
                          gcxx::span<const deviceGraphNode_t> pDependencies =
                            {}) -> deviceGraphNode_t;

  GCXX_FH
  auto AddEventWaitNode(const EventView event,
                        gcxx::span<const deviceGraphNode_t> pDependencies = {})
    -> deviceGraphNode_t;

  GCXX_FH
  auto AddHostNode(const HostNodeParamsView params,
                   gcxx::span<const deviceGraphNode_t> pDependencies = {})
    -> deviceGraphNode_t;

  GCXX_FH
  auto AddKernelNode(const KernelNodeParamsView params,
                     gcxx::span<const deviceGraphNode_t> pDependencies = {})
    -> deviceGraphNode_t;

  GCXX_FH
  auto AddMemFreeNode(void* dptr,
                      gcxx::span<const deviceGraphNode_t> pDependencies = {})
    -> deviceGraphNode_t;

  GCXX_FH
  auto AddMemcpyNode(const Memcpy3DParamsView params,
                     gcxx::span<const deviceGraphNode_t> pDependencies = {})
    -> deviceGraphNode_t;

  GCXX_FH
  auto AddMemcpyNode1D(void* dst, const void* src, std::size_t countBytes,
                       gcxx::span<const deviceGraphNode_t> pDependencies = {})
    -> deviceGraphNode_t;

  GCXX_FH
  auto AddMemsetNode(const MemsetParamsView params,
                     gcxx::span<const deviceGraphNode_t> pDependencies = {})
    -> deviceGraphNode_t;
};

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/graph/graph_view.inl>
#include <gcxx/runtime/details/graph/nodes/child_graph_node_view.inl>

#include <gcxx/macros/undefine_macros.hpp>
#endif
