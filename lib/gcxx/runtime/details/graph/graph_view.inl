// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_GRAPH_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_GRAPH_VIEW_INL_

#include <cstddef>
#include <filesystem>
#include <string_view>


#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/graph/graph_view.hpp>


GCXX_NAMESPACE_MAIN_BEGIN()

struct IfNodeResult {
  GraphView::deviceGraphNode_t conditionalNode{};
  GraphView IfbodyGraph;
};

struct IfElseNodeResult {
  GraphView::deviceGraphNode_t conditionalNode{};
  GraphView IfbodyGraph;
  GraphView ElsebodyGraph;
};

struct WhileNodeResult {
  GraphView::deviceGraphNode_t conditionalNode{};
  GraphView whilebodyGraph;
};

struct SwitchNodeResult {
  GraphView::deviceGraphNode_t conditionalNode;
  std::vector<GraphView> CasesbodyGraph;
};

GCXX_FHC GraphView::GraphView(deviceGraph_t rawgraph) : m_graph(rawgraph) {}

GCXX_FHC auto GraphView::getRawHandle() const -> deviceGraph_t {
  return m_graph;
}

GCXX_FH auto GraphView::SaveDotfile(std::string_view fname,
                                    flags::graphDebugDot flag) const -> void {
  // TODO : Add checks to prevent illegal file name and check folder existance
  const std::string filename{fname};
  driver::graphDebugDotPrint(m_graph, filename.c_str(),
                             static_cast<details_::flag_t>(flag));
}

GCXX_FH auto GraphView::GetNumNodes() const -> size_t {
  return driver::graphGetNumNodes(m_graph);
}

GCXX_FH auto GraphView::GetNumEdges() const -> size_t {
  return driver::graphGetNumEdges(m_graph);
}

GCXX_FH auto GraphView::Clone() const -> GraphView {
  return driver::graphClone(m_graph);
}

// TODO : Need to make better implementation to remove repetions
#if GCXX_CUDA_MODE()
// Create the conditional handle; no default value arg is provided, since i dont
// want the condition value to be undefined at the start of each graph execution
GCXX_FH auto GraphView::CreateConditionalHandle(
  unsigned int defaultLaunchValue,
  flags::graphConditionalHandle flag) -> deviceGraphConditionalHandle_t {
  return driver::graphConditionalHandleCreate(
    m_graph, defaultLaunchValue, static_cast<details_::flag_t>(flag));
}

GCXX_FD
auto GraphView::SetConditional(deviceGraphConditionalHandle_t handle,
                               unsigned int value) -> void {
  driver::graphSetConditional(handle, value);
}

GCXX_FH auto GraphView::AddIfNode(deviceGraphConditionalHandle_t condHandle,
                                  const deviceGraphNode_t* pDependencies,
                                  std::size_t numDependencies) -> IfNodeResult {
  deviceGraphNode_t node          = nullptr;
  deviceGraphNodeParams_t cParams = {
    GCXX_RUNTIME_BACKEND(GraphNodeTypeConditional)};
  cParams.conditional.handle = condHandle;
  cParams.conditional.type   = GCXX_RUNTIME_BACKEND(GraphCondTypeIf);
  cParams.conditional.size   = 1;

  driver::graphAddNode(&node, m_graph, pDependencies, nullptr, numDependencies,
                       &cParams);

  // Extract the body graph from the conditional node parameters
  deviceGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

  return IfNodeResult{node, bodyGraph};
}

GCXX_FH auto GraphView::AddIfElseNode(deviceGraphConditionalHandle_t condHandle,
                                      const deviceGraphNode_t* pDependencies,
                                      std::size_t numDependencies)
  -> IfElseNodeResult {
  deviceGraphNode_t node          = nullptr;
  deviceGraphNodeParams_t cParams = {
    GCXX_RUNTIME_BACKEND(GraphNodeTypeConditional)};
  cParams.conditional.handle = condHandle;
  cParams.conditional.type   = GCXX_RUNTIME_BACKEND(GraphCondTypeIf);
  cParams.conditional.size   = 2;

  driver::graphAddNode(&node, m_graph, pDependencies, nullptr, numDependencies,
                       &cParams);

  // Extract both body graphs from the conditional node parameters
  deviceGraph_t ifBodyGraph   = cParams.conditional.phGraph_out[0];
  deviceGraph_t elseBodyGraph = cParams.conditional.phGraph_out[1];

  return {node, ifBodyGraph, elseBodyGraph};
}

GCXX_FH auto GraphView::AddWhileNode(deviceGraphConditionalHandle_t condHand,
                                     const deviceGraphNode_t* pDependencies,
                                     std::size_t numDependencies)
  -> WhileNodeResult {
  deviceGraphNode_t node          = nullptr;
  deviceGraphNodeParams_t cParams = {
    GCXX_RUNTIME_BACKEND(GraphNodeTypeConditional)};
  cParams.conditional.handle = condHand;
  cParams.conditional.type   = GCXX_RUNTIME_BACKEND(GraphCondTypeWhile);
  cParams.conditional.size   = 1;

  driver::graphAddNode(&node, m_graph, pDependencies, nullptr, numDependencies,
                       &cParams);

  // Extract the body graph from the conditional node parameters
  deviceGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

  return WhileNodeResult{node, GraphView(bodyGraph)};
}

GCXX_FH auto GraphView::AddSwitchNode(
  deviceGraphConditionalHandle_t condHand, std::size_t numCases,
  const deviceGraphNode_t* pDependencies,
  std::size_t numDependencies) -> SwitchNodeResult {
  deviceGraphNode_t node          = nullptr;
  deviceGraphNodeParams_t cParams = {
    GCXX_RUNTIME_BACKEND(GraphNodeTypeConditional)};
  cParams.conditional.handle = condHand;
  cParams.conditional.type   = GCXX_RUNTIME_BACKEND(GraphCondTypeSwitch);
  cParams.conditional.size   = numCases;

  driver::graphAddNode(&node, m_graph, pDependencies, nullptr, numDependencies,
                       &cParams);

  // Extract all case body graphs from the conditional node parameters
  return SwitchNodeResult{node,
                          {&cParams.conditional.phGraph_out[0],
                           &cParams.conditional.phGraph_out[numCases]}};
}
#endif

// ════════════════════════════════════════════════════════════════════════════
// Graph Node Addition Implementations
// ════════════════════════════════════════════════════════════════════════════


GCXX_FH auto GraphView::AddChildGraphNode(
  const GraphView& childGraph, const deviceGraphNode_t* pDependencies,
  std::size_t numDependencies) -> ChildGraphNodeView {
  return driver::graphAddChildGraphNode(m_graph, childGraph.getRawHandle(),
                                        pDependencies, numDependencies);
}

GCXX_FH auto GraphView::AddDependencies(const deviceGraphNode_t* from,
                                        const deviceGraphNode_t* to,
                                        std::size_t numDependencies) -> void {
  driver::graphAddDependencies(m_graph, from, to, nullptr, numDependencies);
}

GCXX_FH auto GraphView::AddEmptyNode(const deviceGraphNode_t* pDependencies,
                                     std::size_t numDependencies)
  -> deviceGraphNode_t {
  return driver::graphAddEmptyNode(m_graph, pDependencies, numDependencies);
}

GCXX_FH auto GraphView::AddEventRecordNode(
  const EventView event, const deviceGraphNode_t* pDependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  return driver::graphAddEventRecordNode(m_graph, event.getRawHandle(),
                                         pDependencies, numDependencies);
}

GCXX_FH auto GraphView::AddEventWaitNode(
  const EventView event, const deviceGraphNode_t* pDependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  return driver::graphAddEventWaitNode(m_graph, event.getRawHandle(),
                                       pDependencies, numDependencies);
}

GCXX_FH auto GraphView::AddHostNode(
  const HostNodeParamsView::deviceHostNodeParams_t* params,
  const deviceGraphNode_t* pDependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  return driver::graphAddHostNode(m_graph, params, pDependencies,
                                  numDependencies);
}

GCXX_FH auto GraphView::AddKernelNode(
  const KernelNodeParamsView::deviceKernelNodeParams_t* params,
  const deviceGraphNode_t* pDependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  return driver::graphAddKernelNode(m_graph, params, pDependencies,
                                    numDependencies);
}

GCXX_FH auto GraphView::AddMemFreeNode(
  void* dptr, const deviceGraphNode_t* pDependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  return driver::graphAddMemFreeNode(m_graph, dptr, pDependencies,
                                     numDependencies);
}

GCXX_FH auto GraphView::AddMemcpyNode(
  const Memcpy3DParamsView::deviceMemcpy3DParams_t* params,
  const deviceGraphNode_t* pDependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  return driver::graphAddMemcpyNode(m_graph, params, pDependencies,
                                    numDependencies);
}

GCXX_FH auto GraphView::AddMemcpyNode1D(
  void* dst, const void* src, std::size_t countBytes,
  const deviceGraphNode_t* pDependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  return driver::graphAddMemcpyNode1D(m_graph, dst, src, countBytes,
                                      pDependencies, numDependencies);
}

GCXX_FH auto GraphView::AddMemsetNode(
  const MemsetParamsView::deviceMemsetParams_t* params,
  const deviceGraphNode_t* pDependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  return driver::graphAddMemsetNode(m_graph, params, pDependencies,
                                    numDependencies);
}

/// CPP STYLE
GCXX_FH auto GraphView::AddChildGraphNode(
  const GraphView& childGraph,
  gcxx::span<const deviceGraphNode_t> pDependencies) -> ChildGraphNodeView {
  return AddChildGraphNode(childGraph, pDependencies.data(),
                           pDependencies.size());
}

GCXX_FH auto GraphView::AddDependencies(
  gcxx::span<const deviceGraphNode_t> from,
  gcxx::span<const deviceGraphNode_t> to) -> void {
  GCXX_RUNTIME_EXPECT(from.size() != to.size(),
                      "Mistamatch in to and from depencey count");
  AddDependencies(from.data(), to.data(), from.size());
}

GCXX_FH auto GraphView::AddEmptyNode(
  gcxx::span<const deviceGraphNode_t> pDependencies) -> deviceGraphNode_t {
  return AddEmptyNode(pDependencies.data(), pDependencies.size());
}

GCXX_FH auto GraphView::AddEventRecordNode(
  const EventView event,
  gcxx::span<const deviceGraphNode_t> pDependencies) -> deviceGraphNode_t {
  return AddEventRecordNode(event, pDependencies.data(), pDependencies.size());
}

GCXX_FH auto GraphView::AddEventWaitNode(
  const EventView event,
  gcxx::span<const deviceGraphNode_t> pDependencies) -> deviceGraphNode_t {
  return AddEventWaitNode(event, pDependencies.data(), pDependencies.size());
}

GCXX_FH auto GraphView::AddHostNode(
  const HostNodeParamsView params,
  gcxx::span<const deviceGraphNode_t> pDependencies) -> deviceGraphNode_t {
  return AddHostNode(&(params.getRawParams()), pDependencies.data(),
                     pDependencies.size());
}

GCXX_FH auto GraphView::AddKernelNode(
  const KernelNodeParamsView params,
  gcxx::span<const deviceGraphNode_t> pDependencies) -> deviceGraphNode_t {
  return AddKernelNode(&(params.getRawParams()), pDependencies.data(),
                       pDependencies.size());
}

GCXX_FH auto GraphView::AddMemFreeNode(
  void* dptr,
  gcxx::span<const deviceGraphNode_t> pDependencies) -> deviceGraphNode_t {
  return AddMemFreeNode(dptr, pDependencies.data(), pDependencies.size());
}

GCXX_FH auto GraphView::AddMemcpyNode(
  const Memcpy3DParamsView params,
  gcxx::span<const deviceGraphNode_t> pDependencies) -> deviceGraphNode_t {
  return AddMemcpyNode(&(params.getRawParams()), pDependencies.data(),
                       pDependencies.size());
}

GCXX_FH auto GraphView::AddMemcpyNode1D(
  void* dst, const void* src, std::size_t countBytes,
  gcxx::span<const deviceGraphNode_t> pDependencies) -> deviceGraphNode_t {
  return AddMemcpyNode1D(dst, src, countBytes, pDependencies.data(),
                         pDependencies.size());
}

GCXX_FH auto GraphView::AddMemsetNode(
  const MemsetParamsView params,
  gcxx::span<const deviceGraphNode_t> pDependencies) -> deviceGraphNode_t {
  return AddMemsetNode(&(params.getRawParams()), pDependencies.data(),
                       pDependencies.size());
}

GCXX_NAMESPACE_MAIN_END()

#endif
