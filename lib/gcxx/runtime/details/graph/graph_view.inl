#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_GRAPH_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_GRAPH_VIEW_INL_

#include <cstddef>
#include <filesystem>
#include <string_view>


#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/graph/graph_view.hpp>


GCXX_NAMESPACE_MAIN_BEGIN

struct IfNodeResult {
  deviceGraphNode_t conditionalNode{};
  GraphView IfbodyGraph;
};

struct IfElseNodeResult {
  deviceGraphNode_t conditionalNode{};
  GraphView IfbodyGraph;
  GraphView ElsebodyGraph;
};

struct WhileNodeResult {
  deviceGraphNode_t conditionalNode{};
  GraphView whilebodyGraph;
};

struct SwitchNodeResult {
  deviceGraphNode_t conditionalNode;
  std::vector<GraphView> CasesbodyGraph;
};

GCXX_FHC GraphView::GraphView(deviceGraph_t rawgraph) : graph_(rawgraph) {}

GCXX_FHC auto GraphView::getRawGraph() const -> deviceGraph_t {
  return graph_;
}

GCXX_FH auto GraphView::SaveDotfile(std::string_view fname,
                                    flags::graphDebugDot flag) const -> void {
  // TODO : Add checks to prevent illegal file name and check folder existance
  const std::string filename{fname};
  GCXX_SAFE_RUNTIME_CALL(GraphDebugDotPrint,
                         "Failed to output the dot file of the graph", graph_,
                         filename.c_str(), static_cast<details_::flag_t>(flag));
}

GCXX_FH auto GraphView::GetNumNodes() const -> size_t {
  size_t numNodes = 0;
  GCXX_SAFE_RUNTIME_CALL(GraphGetNodes, "Failed to get Count of Graph nodes",
                         graph_, nullptr, &numNodes);
  return numNodes;
}

GCXX_FH auto GraphView::GetNumEdges() const -> size_t {
  size_t numEdges = 0;
  GCXX_SAFE_RUNTIME_CALL(GraphGetEdges, "Failed to get count of Graph edges",
                         graph_, nullptr, nullptr,
#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)
                         nullptr,
#endif
                         &numEdges);
  return numEdges;
}

GCXX_FH auto GraphView::Clone() const -> GraphView {
  deviceGraph_t clonedGraph = nullptr;
  GCXX_SAFE_RUNTIME_CALL(GraphClone, "Failed to clone graph", &clonedGraph,
                         graph_);
  return clonedGraph;
}

// TODO : Need to make better implementation to remove repetions
#if GCXX_CUDA_MODE
// Create the conditional handle; no default value arg is provided, since i dont
// want the condition value to be undefined at the start of each graph execution
GCXX_FH auto GraphView::CreateConditionalHandle(
  unsigned int defaultLaunchValue, flags::graphConditionalHandle flag)
  -> deviceGraphConditionalHandle_t {
  deviceGraphConditionalHandle_t out{0};
  GCXX_SAFE_RUNTIME_CALL(GraphConditionalHandleCreate,
                         "Failed to create conditional handle in graph", &out,
                         graph_, defaultLaunchValue,
                         static_cast<details_::flag_t>(flag));
  return out;
}

GCXX_FD auto GraphView::SetConditional(deviceGraphConditionalHandle_t handle,
                                       unsigned int value) -> void {
#if GCXX_CUDA_MODE
  GCXX_RUNTIME_BACKEND(GraphSetConditional)(handle, value);
#elif GCXX_HIP_MODE
#warning "Conditional nodes are not implemented in HIP yet"
#endif
}

GCXX_FH auto GraphView::AddIfNode(deviceGraphConditionalHandle_t condHandle,
                                  const deviceGraphNode_t* pDependencies,
                                  std::size_t numDependencies) -> IfNodeResult {
  deviceGraphNode_t node                    = nullptr;
  details_::deviceGraphNodeParams_t cParams = {
    GCXX_RUNTIME_BACKEND(GraphNodeTypeConditional)};
  cParams.conditional.handle = condHandle;
  cParams.conditional.type   = GCXX_RUNTIME_BACKEND(GraphCondTypeIf);
  cParams.conditional.size   = 1;

  GCXX_SAFE_RUNTIME_CALL(GraphAddNode, "Failed to add If node to graph", &node,
                         graph_, pDependencies,
#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)  // TODO : support dependency data
                         nullptr,
#endif
                         numDependencies, &cParams);

  // Extract the body graph from the conditional node parameters
  deviceGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

  return IfNodeResult{node, GraphView(bodyGraph)};
}

GCXX_FH auto GraphView::AddIfElseNode(deviceGraphConditionalHandle_t condHandle,
                                      const deviceGraphNode_t* pDependencies,
                                      std::size_t numDependencies)
  -> IfElseNodeResult {
  deviceGraphNode_t node                    = nullptr;
  details_::deviceGraphNodeParams_t cParams = {
    GCXX_RUNTIME_BACKEND(GraphNodeTypeConditional)};
  cParams.conditional.handle = condHandle;
  cParams.conditional.type   = GCXX_RUNTIME_BACKEND(GraphCondTypeIf);
  cParams.conditional.size   = 2;

  GCXX_SAFE_RUNTIME_CALL(GraphAddNode, "Failed to add If-Else node to graph",
                         &node, graph_, pDependencies,
#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)  // TODO : support dependency data
                         nullptr,
#endif
                         numDependencies, &cParams);

  // Extract both body graphs from the conditional node parameters
  deviceGraph_t ifBodyGraph   = cParams.conditional.phGraph_out[0];
  deviceGraph_t elseBodyGraph = cParams.conditional.phGraph_out[1];

  return IfElseNodeResult{node, GraphView(ifBodyGraph),
                          GraphView(elseBodyGraph)};
}

GCXX_FH auto GraphView::AddWhileNode(deviceGraphConditionalHandle_t condHand,
                                     const deviceGraphNode_t* pDependencies,
                                     std::size_t numDependencies)
  -> WhileNodeResult {
  deviceGraphNode_t node                    = nullptr;
  details_::deviceGraphNodeParams_t cParams = {
    GCXX_RUNTIME_BACKEND(GraphNodeTypeConditional)};
  cParams.conditional.handle = condHand;
  cParams.conditional.type   = GCXX_RUNTIME_BACKEND(GraphCondTypeWhile);
  cParams.conditional.size   = 1;

  GCXX_SAFE_RUNTIME_CALL(GraphAddNode, "Failed to add While node to graph",
                         &node, graph_, pDependencies,
#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)  // TODO : support dependency data
                         nullptr,
#endif
                         numDependencies, &cParams);

  // Extract the body graph from the conditional node parameters
  deviceGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

  return WhileNodeResult{node, GraphView(bodyGraph)};
}

GCXX_FH auto GraphView::AddSwitchNode(deviceGraphConditionalHandle_t condHand,
                                      std::size_t numCases,
                                      const deviceGraphNode_t* pDependencies,
                                      std::size_t numDependencies)
  -> SwitchNodeResult {
  deviceGraphNode_t node                    = nullptr;
  details_::deviceGraphNodeParams_t cParams = {
    GCXX_RUNTIME_BACKEND(GraphNodeTypeConditional)};
  cParams.conditional.handle = condHand;
  cParams.conditional.type   = GCXX_RUNTIME_BACKEND(GraphCondTypeSwitch);
  cParams.conditional.size   = numCases;

  GCXX_SAFE_RUNTIME_CALL(GraphAddNode, "Failed to add Switch node to graph",
                         &node, graph_, pDependencies,
#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)  // TODO : support dependency data
                         nullptr,
#endif
                         numDependencies, &cParams);

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
  deviceGraphNode_t node = nullptr;
  GCXX_SAFE_RUNTIME_CALL(
    GraphAddChildGraphNode, "Failed to Add Child graph Node to Graph", &node,
    graph_, pDependencies, numDependencies, childGraph.getRawGraph());
  return node;
}

GCXX_FH auto GraphView::AddDependencies(const deviceGraphNode_t* from,
                                        const deviceGraphNode_t* to,
                                        std::size_t numDependencies) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphAddDependencies,
                         "Failed to Add Dependency between graph Nodes", graph_,
                         from, to,
#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)  // TODO : support dependency data
                         nullptr,
#endif
                         numDependencies);
}

GCXX_FH auto GraphView::AddEmptyNode(const deviceGraphNode_t* pDependencies,
                                     std::size_t numDependencies)
  -> deviceGraphNode_t {
  deviceGraphNode_t node = nullptr;
  GCXX_SAFE_RUNTIME_CALL(GraphAddEmptyNode, "Failed to Add Empty Node to Graph",
                         &node, graph_, pDependencies, numDependencies);
  return node;
}

GCXX_FH auto GraphView::AddEventRecordNode(
  const EventView event, const deviceGraphNode_t* pDependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  deviceGraphNode_t node = nullptr;
  GCXX_SAFE_RUNTIME_CALL(
    GraphAddEventRecordNode, "Failed to Add Event record Node to Graph", &node,
    graph_, pDependencies, numDependencies, event.getRawEvent());
  return node;
}

GCXX_FH auto GraphView::AddEventWaitNode(const EventView event,
                                         const deviceGraphNode_t* pDependencies,
                                         std::size_t numDependencies)
  -> deviceGraphNode_t {
  deviceGraphNode_t node = nullptr;
  GCXX_SAFE_RUNTIME_CALL(
    GraphAddEventWaitNode, "Failed to Add Event Wait Node to Graph", &node,
    graph_, pDependencies, numDependencies, event.getRawEvent());
  return node;
}

GCXX_FH auto GraphView::AddHostNode(const deviceHostNodeParams_t* params,
                                    const deviceGraphNode_t* pDependencies,
                                    std::size_t numDependencies)
  -> deviceGraphNode_t {
  deviceGraphNode_t node = nullptr;
  GCXX_SAFE_RUNTIME_CALL(GraphAddHostNode, "Failed to Add Host Node to Graph",
                         &node, graph_, pDependencies, numDependencies, params);
  return node;
}

GCXX_FH auto GraphView::AddKernelNode(const deviceKernelNodeParams_t* params,
                                      const deviceGraphNode_t* pDependencies,
                                      std::size_t numDependencies)
  -> deviceGraphNode_t {
  deviceGraphNode_t node = nullptr;
  GCXX_SAFE_RUNTIME_CALL(GraphAddKernelNode,
                         "Failed to Add Kernel Node to Graph", &node, graph_,
                         pDependencies, numDependencies, params);
  return node;
}

GCXX_FH auto GraphView::AddMemFreeNode(void* dptr,
                                       const deviceGraphNode_t* pDependencies,
                                       std::size_t numDependencies)
  -> deviceGraphNode_t {
  deviceGraphNode_t node = nullptr;
  GCXX_SAFE_RUNTIME_CALL(GraphAddMemFreeNode,
                         "Failed to Add Mem free Node to Graph", &node, graph_,
                         pDependencies, numDependencies, dptr);
  return node;
}

GCXX_FH auto GraphView::AddMemcpyNode(const deviceMemcpy3DParams_t* params,
                                      const deviceGraphNode_t* pDependencies,
                                      std::size_t numDependencies)
  -> deviceGraphNode_t {
  deviceGraphNode_t node = nullptr;
  GCXX_SAFE_RUNTIME_CALL(GraphAddMemcpyNode,
                         "Failed to Add Memcpy Node to Graph", &node, graph_,
                         pDependencies, numDependencies, params);
  return node;
}

GCXX_FH auto GraphView::AddMemcpyNode1D(void* dst, const void* src,
                                        std::size_t countBytes,
                                        const deviceGraphNode_t* pDependencies,
                                        std::size_t numDependencies)
  -> deviceGraphNode_t {
  deviceGraphNode_t node = nullptr;
  GCXX_SAFE_RUNTIME_CALL(GraphAddMemcpyNode1D,
                         "Failed to Add Memcpy1D Node to Graph", &node, graph_,
                         pDependencies, numDependencies, dst, src, countBytes,
                         GCXX_RUNTIME_BACKEND(MemcpyDefault));
  return node;
}

GCXX_FH auto GraphView::AddMemsetNode(const deviceMemsetParams_t* params,
                                      const deviceGraphNode_t* pDependencies,
                                      std::size_t numDependencies)
  -> deviceGraphNode_t {
  deviceGraphNode_t node = nullptr;
  GCXX_SAFE_RUNTIME_CALL(GraphAddMemsetNode,
                         "Failed to Add Memset Node to Graph", &node, graph_,
                         pDependencies, numDependencies, params);
  return node;
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
  // TODO : add asserts
  AddDependencies(from.data(), to.data(), from.size());
}

GCXX_FH auto GraphView::AddEmptyNode(
  gcxx::span<const deviceGraphNode_t> pDependencies) -> deviceGraphNode_t {
  return AddEmptyNode(pDependencies.data(), pDependencies.size());
}

GCXX_FH auto GraphView::AddEventRecordNode(
  const EventView event, gcxx::span<const deviceGraphNode_t> pDependencies)
  -> deviceGraphNode_t {
  return AddEventRecordNode(event, pDependencies.data(), pDependencies.size());
}

GCXX_FH auto GraphView::AddEventWaitNode(
  const EventView event, gcxx::span<const deviceGraphNode_t> pDependencies)
  -> deviceGraphNode_t {
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
  void* dptr, gcxx::span<const deviceGraphNode_t> pDependencies)
  -> deviceGraphNode_t {
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

GCXX_NAMESPACE_MAIN_END

#include <gcxx/macros/undefine_macros.hpp>

#endif
