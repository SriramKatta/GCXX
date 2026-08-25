// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_GRAPH_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_GRAPH_VIEW_INL_

#include <cstddef>
#include <filesystem>
#include <string_view>
#include <utility>
#include <vector>


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

GCXX_FH auto GraphView::saveDotfile(std::string_view fname,
                                    flags::graphDebugDot flag) const -> void {
  GCXX_RUNTIME_EXPECT(!fname.empty(),
                      "GraphView::saveDotfile: file name must not be empty");
  GCXX_RUNTIME_EXPECT(
    fname.find('\0') == std::string_view::npos,
    "GraphView::saveDotfile: embedded null byte in file name");

  const std::filesystem::path path{fname};
  const std::filesystem::path parentDir = path.parent_path();
  GCXX_RUNTIME_EXPECT(
    parentDir.empty() || std::filesystem::is_directory(parentDir),
    "GraphView::saveDotfile: parent directory does not exist");
  GCXX_RUNTIME_EXPECT(!std::filesystem::is_directory(path),
                      "GraphView::saveDotfile: path refers to a directory");

  const std::string filename{fname};
  driver::graphDebugDotPrint(m_graph, filename.c_str(),
                             static_cast<details_::flag_t>(flag));
}

GCXX_FH auto GraphView::getNumNodes() const -> size_t {
  return driver::graphGetNumNodes(m_graph);
}

GCXX_FH auto GraphView::getNumEdges() const -> size_t {
  return driver::graphGetNumEdges(m_graph);
}

GCXX_FH auto GraphView::clone() const -> GraphView {
  return driver::graphClone(m_graph);
}

#if GCXX_CUDA_MODE()
GCXX_FH auto GraphView::addConditionalNode(
  flags::graphConditionalNode condType, std::size_t numBodyGraphs,
  deviceGraphConditionalHandle_t condHandle,
  gcxx::span<const GraphNodeView> pDependencies)
  -> std::pair<deviceGraphNode_t, std::vector<deviceGraph_t>> {
  const auto rawDependencies = toRawDependencies(pDependencies);

  deviceGraphNode_t node          = nullptr;
  deviceGraphNodeParams_t cParams = {
    GCXX_RUNTIME_BACKEND(GraphNodeTypeConditional)};
  // phGraph_out stays null: the driver fills it in during node creation.
  cParams.conditional.handle = condHandle;
  cParams.conditional.type   = static_cast<decltype(cParams.conditional.type)>(
    static_cast<details_::flag_t>(condType));
  cParams.conditional.size = static_cast<unsigned int>(numBodyGraphs);

  driver::graphAddNode(&node, m_graph, rawDependencies.data(), nullptr,
                       rawDependencies.size(), &cParams);

  std::vector<deviceGraph_t> bodyGraphs(
    cParams.conditional.phGraph_out,
    cParams.conditional.phGraph_out + numBodyGraphs);
  return {node, std::move(bodyGraphs)};
}

// Create the conditional handle; no default value arg is provided, since i dont
// want the condition value to be undefined at the start of each graph execution
GCXX_FH auto GraphView::createConditionalHandle(
  unsigned int defaultLaunchValue,
  flags::graphConditionalHandle flag) -> deviceGraphConditionalHandle_t {
  return driver::graphConditionalHandleCreate(
    m_graph, defaultLaunchValue, static_cast<details_::flag_t>(flag));
}

GCXX_FD
auto GraphView::setConditional(deviceGraphConditionalHandle_t handle,
                               unsigned int value) -> void {
  driver::graphSetConditional(handle, value);
}

GCXX_FH auto GraphView::addIfNode(deviceGraphConditionalHandle_t condHandle,
                                  gcxx::span<const GraphNodeView> pDependencies)
  -> IfNodeResult {
  auto [node, bodyGraphs] = addConditionalNode(flags::graphConditionalNode::If,
                                               1, condHandle, pDependencies);
  return IfNodeResult{node, bodyGraphs[0]};
}

GCXX_FH auto GraphView::addIfElseNode(
  deviceGraphConditionalHandle_t condHandle,
  gcxx::span<const GraphNodeView> pDependencies) -> IfElseNodeResult {
  auto [node, bodyGraphs] = addConditionalNode(flags::graphConditionalNode::If,
                                               2, condHandle, pDependencies);
  return IfElseNodeResult{node, bodyGraphs[0], bodyGraphs[1]};
}

GCXX_FH auto GraphView::addWhileNode(
  deviceGraphConditionalHandle_t condHand,
  gcxx::span<const GraphNodeView> pDependencies) -> WhileNodeResult {
  auto [node, bodyGraphs] = addConditionalNode(
    flags::graphConditionalNode::While, 1, condHand, pDependencies);
  return WhileNodeResult{node, bodyGraphs[0]};
}

GCXX_FH auto GraphView::addSwitchNode(
  deviceGraphConditionalHandle_t condHand, std::size_t numCases,
  gcxx::span<const GraphNodeView> pDependencies) -> SwitchNodeResult {
  auto [node, caseGraphs] = addConditionalNode(
    flags::graphConditionalNode::Switch, numCases, condHand, pDependencies);
  return SwitchNodeResult{node, {caseGraphs.begin(), caseGraphs.end()}};
}
#endif

// ════════════════════════════════════════════════════════════════════════════
// Generic Node Addition Implementations
// ════════════════════════════════════════════════════════════════════════════

GCXX_NAMESPACE_DETAILS_BEGIN()

// Union-member fill helpers. CUDA's cudaGraphNodeParams union members for
// kernel/memset/host/alloc/ext-sem are the V2 structs while HIP's (and the
// types our params wrap) are the V1 structs; the field lists are identical,
// so generic field copies work on both backends. The remaining kinds
// (memcpy/event/free/graph) use the exact same struct type in the union and
// are assigned directly at the call site.

template <typename MemberT>
GCXX_FH auto fillKernelMember(
  MemberT& member, const driver::deviceKernelNodeParams_t& src) -> void {
  member.func           = src.func;
  member.gridDim        = src.gridDim;
  member.blockDim       = src.blockDim;
  member.sharedMemBytes = src.sharedMemBytes;
  member.kernelParams   = src.kernelParams;
  member.extra          = src.extra;
}

template <typename MemberT>
GCXX_FH auto fillMemsetMember(MemberT& member,
                              const driver::deviceMemsetParams_t& src) -> void {
  member.dst         = src.dst;
  member.pitch       = src.pitch;
  member.value       = src.value;
  member.elementSize = src.elementSize;
  member.width       = src.width;
  member.height      = src.height;
}

template <typename MemberT>
GCXX_FH auto fillHostMember(MemberT& member,
                            const driver::deviceHostNodeParams_t& src) -> void {
  member.fn       = src.fn;
  member.userData = src.userData;
}

template <typename MemberT>
GCXX_FH auto fillMemAllocMember(
  MemberT& member, const driver::deviceMemAllocNodeParams_t& src) -> void {
  member.poolProps       = src.poolProps;
  member.accessDescs     = src.accessDescs;
  member.accessDescCount = src.accessDescCount;
  member.bytesize        = src.bytesize;
  // src.dptr (the allocation output) is intentionally not copied: the driver
  // fills the union member during creation; read it back through
  // MemAllocNodeView::getParams().
}

template <typename MemberT, typename SrcT>
GCXX_FH auto fillExtSemMember(MemberT& member, const SrcT& src) -> void {
  member.extSemArray = src.extSemArray;
  member.paramsArray = src.paramsArray;
  member.numExtSems  = src.numExtSems;
}

GCXX_NAMESPACE_DETAILS_END()

// needed to convert from individual node types to raw nodes since it passed to
// backend driver functions
GCXX_FH auto GraphView::toRawDependencies(
  gcxx::span<const GraphNodeView> pDependencies)
  -> std::vector<deviceGraphNode_t> {
  std::vector<deviceGraphNode_t> rawDependencies;
  rawDependencies.reserve(pDependencies.size());
  for (std::size_t i = 0; i < pDependencies.size(); ++i) {
    rawDependencies.push_back(pDependencies[i].getRawHandle());
  }
  return rawDependencies;
}

GCXX_FH auto GraphView::addNode(gcxx::span<const GraphNodeView> pDependencies)
  -> GraphNodeView {
  const auto rawDependencies = toRawDependencies(pDependencies);

  deviceGraphNodeParams_t nodeParams{};
  nodeParams.type = GCXX_RUNTIME_BACKEND(GraphNodeTypeEmpty);
  deviceGraphNode_t node{driver::INVALID_GRAPH_NODE};
  driver::graphAddNode(&node, m_graph, rawDependencies.data(), nullptr,
                       rawDependencies.size(), &nodeParams);
  return GraphNodeView{node};
}

GCXX_FH auto GraphView::addNode(const KernelNodeParamsView& params,
                                gcxx::span<const GraphNodeView> pDependencies)
  -> KernelNodeView {
  const auto rawDependencies = toRawDependencies(pDependencies);

  deviceGraphNodeParams_t nodeParams{};
  nodeParams.type = GCXX_RUNTIME_BACKEND(GraphNodeTypeKernel);
  details_::fillKernelMember(nodeParams.kernel, params.getRawParams());
  deviceGraphNode_t node{driver::INVALID_GRAPH_NODE};
  driver::graphAddNode(&node, m_graph, rawDependencies.data(), nullptr,
                       rawDependencies.size(), &nodeParams);
  return KernelNodeView{node};
}

GCXX_FH auto GraphView::addNode(const Memcpy3DParamsView& params,
                                gcxx::span<const GraphNodeView> pDependencies)
  -> MemcpyNodeView {
  const auto rawDependencies = toRawDependencies(pDependencies);

  deviceGraphNodeParams_t nodeParams{};
  nodeParams.type = GCXX_RUNTIME_BACKEND(GraphNodeTypeMemcpy);
  // flags/reserved stay zeroed by the value-initialization above.
  nodeParams.memcpy.copyParams = params.getRawParams();
  deviceGraphNode_t node{driver::INVALID_GRAPH_NODE};
  driver::graphAddNode(&node, m_graph, rawDependencies.data(), nullptr,
                       rawDependencies.size(), &nodeParams);
  return MemcpyNodeView{node};
}

GCXX_FH auto GraphView::addNode(const MemsetParamsView& params,
                                gcxx::span<const GraphNodeView> pDependencies)
  -> MemsetNodeView {
  const auto rawDependencies = toRawDependencies(pDependencies);

  deviceGraphNodeParams_t nodeParams{};
  nodeParams.type = GCXX_RUNTIME_BACKEND(GraphNodeTypeMemset);
  details_::fillMemsetMember(nodeParams.memset, params.getRawParams());
  deviceGraphNode_t node{driver::INVALID_GRAPH_NODE};
  driver::graphAddNode(&node, m_graph, rawDependencies.data(), nullptr,
                       rawDependencies.size(), &nodeParams);
  return MemsetNodeView{node};
}

GCXX_FH auto GraphView::addNode(const HostNodeParamsView& params,
                                gcxx::span<const GraphNodeView> pDependencies)
  -> HostNodeView {
  const auto rawDependencies = toRawDependencies(pDependencies);

  deviceGraphNodeParams_t nodeParams{};
  nodeParams.type = GCXX_RUNTIME_BACKEND(GraphNodeTypeHost);
  details_::fillHostMember(nodeParams.host, params.getRawParams());
  deviceGraphNode_t node{driver::INVALID_GRAPH_NODE};
  driver::graphAddNode(&node, m_graph, rawDependencies.data(), nullptr,
                       rawDependencies.size(), &nodeParams);
  return HostNodeView{node};
}

GCXX_FH auto GraphView::addNode(const EventRecordNodeParamsView& params,
                                gcxx::span<const GraphNodeView> pDependencies)
  -> EventRecordNodeView {
  const auto rawDependencies = toRawDependencies(pDependencies);

  deviceGraphNodeParams_t nodeParams{};
  nodeParams.type        = GCXX_RUNTIME_BACKEND(GraphNodeTypeEventRecord);
  nodeParams.eventRecord = params.getRawParams();
  deviceGraphNode_t node{driver::INVALID_GRAPH_NODE};
  driver::graphAddNode(&node, m_graph, rawDependencies.data(), nullptr,
                       rawDependencies.size(), &nodeParams);
  return EventRecordNodeView{node};
}

GCXX_FH auto GraphView::addNode(const EventWaitNodeParamsView& params,
                                gcxx::span<const GraphNodeView> pDependencies)
  -> EventWaitNodeView {
  const auto rawDependencies = toRawDependencies(pDependencies);

  deviceGraphNodeParams_t nodeParams{};
  nodeParams.type      = GCXX_RUNTIME_BACKEND(GraphNodeTypeWaitEvent);
  nodeParams.eventWait = params.getRawParams();
  deviceGraphNode_t node{driver::INVALID_GRAPH_NODE};
  driver::graphAddNode(&node, m_graph, rawDependencies.data(), nullptr,
                       rawDependencies.size(), &nodeParams);
  return EventWaitNodeView{node};
}

GCXX_FH auto GraphView::addNode(const MemFreeNodeParamsView& params,
                                gcxx::span<const GraphNodeView> pDependencies)
  -> MemFreeNodeView {
  const auto rawDependencies = toRawDependencies(pDependencies);

  deviceGraphNodeParams_t nodeParams{};
  nodeParams.type = GCXX_RUNTIME_BACKEND(GraphNodeTypeMemFree);
  nodeParams.free = params.getRawParams();
  deviceGraphNode_t node{driver::INVALID_GRAPH_NODE};
  driver::graphAddNode(&node, m_graph, rawDependencies.data(), nullptr,
                       rawDependencies.size(), &nodeParams);
  return MemFreeNodeView{node};
}

GCXX_FH auto GraphView::addNode(const ChildGraphNodeParamsView& params,
                                gcxx::span<const GraphNodeView> pDependencies)
  -> ChildGraphNodeView {
  const auto rawDependencies = toRawDependencies(pDependencies);

  deviceGraphNodeParams_t nodeParams{};
  nodeParams.type  = GCXX_RUNTIME_BACKEND(GraphNodeTypeGraph);
  nodeParams.graph = params.getRawParams();
  deviceGraphNode_t node{driver::INVALID_GRAPH_NODE};
  driver::graphAddNode(&node, m_graph, rawDependencies.data(), nullptr,
                       rawDependencies.size(), &nodeParams);
  return ChildGraphNodeView{node};
}

GCXX_FH auto GraphView::addNode(const MemAllocNodeParamsView& params,
                                gcxx::span<const GraphNodeView> pDependencies)
  -> MemAllocNodeView {
  const auto rawDependencies = toRawDependencies(pDependencies);

  deviceGraphNodeParams_t nodeParams{};
  nodeParams.type = GCXX_RUNTIME_BACKEND(GraphNodeTypeMemAlloc);
  details_::fillMemAllocMember(nodeParams.alloc, params.getRawParams());
  deviceGraphNode_t node{driver::INVALID_GRAPH_NODE};
  driver::graphAddNode(&node, m_graph, rawDependencies.data(), nullptr,
                       rawDependencies.size(), &nodeParams);
  return MemAllocNodeView{node};
}

GCXX_FH auto GraphView::addNode(
  const ExternalSemaphoreSignalNodeParamsView& params,
  gcxx::span<const GraphNodeView> pDependencies)
  -> ExternalSemaphoreSignalNodeView {
  const auto rawDependencies = toRawDependencies(pDependencies);

  deviceGraphNodeParams_t nodeParams{};
  nodeParams.type = GCXX_RUNTIME_BACKEND(GraphNodeTypeExtSemaphoreSignal);
  details_::fillExtSemMember(nodeParams.extSemSignal, params.getRawParams());
  deviceGraphNode_t node{driver::INVALID_GRAPH_NODE};
  driver::graphAddNode(&node, m_graph, rawDependencies.data(), nullptr,
                       rawDependencies.size(), &nodeParams);
  return ExternalSemaphoreSignalNodeView{node};
}

GCXX_FH auto GraphView::addNode(
  const ExternalSemaphoreWaitNodeParamsView& params,
  gcxx::span<const GraphNodeView> pDependencies)
  -> ExternalSemaphoreWaitNodeView {
  const auto rawDependencies = toRawDependencies(pDependencies);

  deviceGraphNodeParams_t nodeParams{};
  nodeParams.type = GCXX_RUNTIME_BACKEND(GraphNodeTypeExtSemaphoreWait);
  details_::fillExtSemMember(nodeParams.extSemWait, params.getRawParams());
  deviceGraphNode_t node{driver::INVALID_GRAPH_NODE};
  driver::graphAddNode(&node, m_graph, rawDependencies.data(), nullptr,
                       rawDependencies.size(), &nodeParams);
  return ExternalSemaphoreWaitNodeView{node};
}

GCXX_FH auto GraphView::addDependencies(gcxx::span<const GraphNodeView> from,
                                        gcxx::span<const GraphNodeView> to)
  -> void {
  GCXX_RUNTIME_EXPECT(from.size() == to.size(),
                      "Mismatch in to and from dependency count");
  const auto rawFrom = toRawDependencies(from);
  const auto rawTo   = toRawDependencies(to);
  driver::graphAddDependencies(m_graph, rawFrom.data(), rawTo.data(), nullptr,
                               rawFrom.size());
}

GCXX_NAMESPACE_MAIN_END()

#endif
