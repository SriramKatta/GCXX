// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_GRAPH_VIEW_HPP_
#define GCXX_RUNTIME_GRAPH_GRAPH_VIEW_HPP_

#include <cstddef>
#include <string_view>
#include <utility>
#include <vector>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/flags/graph_flags.hpp>
#include <gcxx/runtime/graph/graph_params.hpp>
#include <gcxx/runtime/memory/spans/spans.hpp>
#include <gcxx/runtime_backend/backend_graph.hpp>

#include <gcxx/runtime/graph/graph_nodes.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


class GraphView;
// Result struct for addIfNode with named fields.
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
  using raw_handle_type = driver::deviceGraph_t;

  GCXX_FHC GraphView() = default;
  GCXX_FHC GraphView(deviceGraph_t rawgraph);
  GCXX_FHC auto getRawHandle() const -> deviceGraph_t;
  GCXX_FH auto saveDotfile(std::string_view,
                           flags::graphDebugDot) const -> void;
  GCXX_FH auto getNumNodes() const -> size_t;
  GCXX_FH auto getNumEdges() const -> size_t;
  GCXX_FH auto clone() const -> GraphView;

#if GCXX_CUDA_MODE()
  GCXX_FH auto createConditionalHandle(
    unsigned int defaultLaunchValue,
    flags::graphConditionalHandle flag = flags::graphConditionalHandle::None)
    -> deviceGraphConditionalHandle_t;

  GCXX_FD
  static auto setConditional(deviceGraphConditionalHandle_t,
                             unsigned int) -> void;

  [[nodiscard]] GCXX_FH auto addIfNode(
    deviceGraphConditionalHandle_t condHand,
    gcxx::span<const GraphNodeView> pDependencies = {}) -> IfNodeResult;

  [[nodiscard]] GCXX_FH auto addIfElseNode(
    deviceGraphConditionalHandle_t condHand,
    gcxx::span<const GraphNodeView> pDependencies = {}) -> IfElseNodeResult;

  [[nodiscard]] GCXX_FH auto addWhileNode(
    deviceGraphConditionalHandle_t condHand,
    gcxx::span<const GraphNodeView> pDependencies = {}) -> WhileNodeResult;

  [[nodiscard]] GCXX_FH auto addSwitchNode(
    deviceGraphConditionalHandle_t condHand, std::size_t numCases,
    gcxx::span<const GraphNodeView> pDependencies = {}) -> SwitchNodeResult;
#endif
  // ─── Generic node addition ────────────────────────────────────────────────
  // addNode dispatches on the payload type and returns the matching typed
  // node view. Every kind is created through the union-based
  // driver::graphAddNode, so there are no per-kind add methods. Dependencies
  // are any GraphNodeView-derived views; omit them for a root node.

  // Empty (no-op) node.
  [[nodiscard]] GCXX_FH auto addNode(
    gcxx::span<const GraphNodeView> pDependencies = {}) -> GraphNodeView;

  [[nodiscard]] GCXX_FH auto addNode(
    const KernelNodeParamsView& params,
    gcxx::span<const GraphNodeView> pDependencies = {}) -> KernelNodeView;

  [[nodiscard]] GCXX_FH auto addNode(
    const Memcpy3DParamsView& params,
    gcxx::span<const GraphNodeView> pDependencies = {}) -> MemcpyNodeView;

  [[nodiscard]] GCXX_FH auto addNode(
    const MemsetParamsView& params,
    gcxx::span<const GraphNodeView> pDependencies = {}) -> MemsetNodeView;

  [[nodiscard]] GCXX_FH auto addNode(
    const HostNodeParamsView& params,
    gcxx::span<const GraphNodeView> pDependencies = {}) -> HostNodeView;

  [[nodiscard]] GCXX_FH auto addNode(
    const EventRecordNodeParamsView& params,
    gcxx::span<const GraphNodeView> pDependencies = {}) -> EventRecordNodeView;

  [[nodiscard]] GCXX_FH auto addNode(
    const EventWaitNodeParamsView& params,
    gcxx::span<const GraphNodeView> pDependencies = {}) -> EventWaitNodeView;

  [[nodiscard]] GCXX_FH auto addNode(
    const MemFreeNodeParamsView& params,
    gcxx::span<const GraphNodeView> pDependencies = {}) -> MemFreeNodeView;

  [[nodiscard]] GCXX_FH auto addNode(
    const ChildGraphNodeParamsView& params,
    gcxx::span<const GraphNodeView> pDependencies = {}) -> ChildGraphNodeView;

  [[nodiscard]] GCXX_FH auto addNode(
    const MemAllocNodeParamsView& params,
    gcxx::span<const GraphNodeView> pDependencies = {}) -> MemAllocNodeView;

  [[nodiscard]] GCXX_FH auto addNode(
    const ExternalSemaphoreSignalNodeParamsView& params,
    gcxx::span<const GraphNodeView> pDependencies = {})
    -> ExternalSemaphoreSignalNodeView;

  [[nodiscard]] GCXX_FH auto addNode(
    const ExternalSemaphoreWaitNodeParamsView& params,
    gcxx::span<const GraphNodeView> pDependencies = {})
    -> ExternalSemaphoreWaitNodeView;

  GCXX_FH auto addDependencies(gcxx::span<const GraphNodeView> from,
                               gcxx::span<const GraphNodeView> to) -> void;

 protected:
  deviceGraph_t m_graph{driver::INVALID_GRAPH};  // NOLINT

 private:
#if GCXX_CUDA_MODE()
  GCXX_FH auto addConditionalNode(flags::graphConditionalNode condType,
                                  std::size_t numBodyGraphs,
                                  deviceGraphConditionalHandle_t condHandle,
                                  gcxx::span<const GraphNodeView> pDependencies)
    -> std::pair<deviceGraphNode_t, std::vector<deviceGraph_t>>;
#endif
  GCXX_FH static auto toRawDependencies(
    gcxx::span<const GraphNodeView> pDependencies)
    -> std::vector<deviceGraphNode_t>;
};

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/graph/graph_view.inl>
#include <gcxx/runtime/details/graph/nodes/child_graph_node_view.inl>
#include <gcxx/runtime/details/graph/params/graph_child_graph_node_params.inl>


#endif
