#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_VIEW_INL_

#include <cstddef>
#include <filesystem>
#include <string_view>


#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/graph/graph_view.hpp>
#include <gcxx/runtime/runtime_error.hpp>


GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FHC GraphView::GraphView(deviceGraph_t rawgraph) : graph_(rawgraph) {}

GCXX_FHC auto GraphView::getRawGraph() const -> deviceGraph_t {
  return graph_;
}

GCXX_FH auto GraphView::SaveDotfile(std::string_view fname,
                                    flags::graphDebugDot flag) const -> void {
  // TODO : Add checks to prevent illegal file name and check folder existance
  GCXX_SAFE_RUNTIME_CALL(GraphDebugDotPrint,
                         "Failed to output the dot file of the graph", graph_,
                         fname.data(), static_cast<flag_t>(flag));
}

GCXX_FH auto GraphView::GetNumNodes() const -> size_t {
  size_t numNodes = 0;
  GCXX_SAFE_RUNTIME_CALL(GraphGetNodes, "Failed to get graph nodes", graph_,
                         nullptr, &numNodes);
  return numNodes;
}

GCXX_FH auto GraphView::GetNumEdges() const -> size_t {
  size_t numEdges = 0;
  GCXX_SAFE_RUNTIME_CALL(GraphGetEdges, "Failed to get graph edges", graph_,
                         nullptr, nullptr, &numEdges);
  return numEdges;
}

GCXX_FH auto GraphView::Clone() const -> GraphView {
  deviceGraph_t clonedGraph;
  GCXX_SAFE_RUNTIME_CALL(GraphClone, "Failed to clone graph", &clonedGraph,
                         graph_);
  return clonedGraph;
}

// ════════════════════════════════════════════════════════════════════════════
// Graph Node Addition Implementations
// ════════════════════════════════════════════════════════════════════════════

GCXX_FH auto GraphView::AddChildGraphNode(
  const deviceGraphNode_t* pDependencies, size_t numDependencies,
  const GraphView& childGraph) -> deviceGraphNode_t {
  deviceGraphNode_t node;
  GCXX_SAFE_RUNTIME_CALL(
    GraphAddChildGraphNode, "Failed to add child graph node to graph", &node,
    graph_, pDependencies, numDependencies, childGraph.getRawGraph());
  return node;
}

GCXX_FH auto GraphView::AddDependencies(const deviceGraphNode_t* from,
                                        const deviceGraphNode_t* to,
                                        size_t numDependencies) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphAddDependencies,
                         "Failed to add dependencies to graph", graph_, from,
                         to, numDependencies);
}

GCXX_FH auto GraphView::AddEmptyNode(const deviceGraphNode_t* pDependencies,
                                     size_t numDependencies)
  -> deviceGraphNode_t {
  deviceGraphNode_t node;
  GCXX_SAFE_RUNTIME_CALL(GraphAddEmptyNode, "Failed to add empty node to graph",
                         &node, graph_, pDependencies, numDependencies);
  return node;
}

GCXX_FH auto GraphView::AddEventRecordNode(
  const deviceGraphNode_t* pDependencies, size_t numDependencies,
  deviceEvent_t event) -> deviceGraphNode_t {
  deviceGraphNode_t node;
  GCXX_SAFE_RUNTIME_CALL(GraphAddEventRecordNode,
                         "Failed to add event record node to graph", &node,
                         graph_, pDependencies, numDependencies, event);
  return node;
}

GCXX_FH auto GraphView::AddEventWaitNode(const deviceGraphNode_t* pDependencies,
                                         size_t numDependencies,
                                         deviceEvent_t event)
  -> deviceGraphNode_t {
  deviceGraphNode_t node;
  GCXX_SAFE_RUNTIME_CALL(GraphAddEventWaitNode,
                         "Failed to add event wait node to graph", &node,
                         graph_, pDependencies, numDependencies, event);
  return node;
}

GCXX_FH auto GraphView::AddExternalSemaphoresSignalNode(
  const deviceGraphNode_t* pDependencies, size_t numDependencies,
  const details_::deviceExternalSemaphoreSignalNodeParams_t* nodeParams)
  -> deviceGraphNode_t {
  deviceGraphNode_t node;
  GCXX_SAFE_RUNTIME_CALL(
    GraphAddExternalSemaphoresSignalNode,
    "Failed to add external semaphores signal node to graph", &node, graph_,
    pDependencies, numDependencies, nodeParams);
  return node;
}

GCXX_FH auto GraphView::AddExternalSemaphoresWaitNode(
  const deviceGraphNode_t* pDependencies, size_t numDependencies,
  const details_::deviceExternalSemaphoreWaitNodeParams_t* nodeParams)
  -> deviceGraphNode_t {
  deviceGraphNode_t node;
  GCXX_SAFE_RUNTIME_CALL(GraphAddExternalSemaphoresWaitNode,
                         "Failed to add external semaphores wait node to graph",
                         &node, graph_, pDependencies, numDependencies,
                         nodeParams);
  return node;
}

GCXX_FH auto GraphView::AddHostNode(
  const deviceGraphNode_t* pDependencies, size_t numDependencies,
  const details_::deviceHostNodeParams_t* nodeParams) -> deviceGraphNode_t {
  deviceGraphNode_t node;
  GCXX_SAFE_RUNTIME_CALL(GraphAddHostNode, "Failed to add host node to graph",
                         &node, graph_, pDependencies, numDependencies,
                         nodeParams);
  return node;
}

GCXX_FH auto GraphView::AddKernelNode(
  const deviceGraphNode_t* pDependencies, size_t numDependencies,
  const details_::deviceKernelNodeParams_t* nodeParams) -> deviceGraphNode_t {
  deviceGraphNode_t node;
  GCXX_SAFE_RUNTIME_CALL(GraphAddKernelNode,
                         "Failed to add kernel node to graph", &node, graph_,
                         pDependencies, numDependencies, nodeParams);
  return node;
}

GCXX_FH auto GraphView::AddMemAllocNode(
  const deviceGraphNode_t* pDependencies, size_t numDependencies,
  details_::deviceMemAllocNodeParams_t* nodeParams) -> deviceGraphNode_t {
  deviceGraphNode_t node;
  GCXX_SAFE_RUNTIME_CALL(GraphAddMemAllocNode,
                         "Failed to add memory allocation node to graph", &node,
                         graph_, pDependencies, numDependencies, nodeParams);
  return node;
}

GCXX_FH auto GraphView::AddMemFreeNode(const deviceGraphNode_t* pDependencies,
                                       size_t numDependencies, void* dptr)
  -> deviceGraphNode_t {
  deviceGraphNode_t node;
  GCXX_SAFE_RUNTIME_CALL(GraphAddMemFreeNode,
                         "Failed to add memory free node to graph", &node,
                         graph_, pDependencies, numDependencies, dptr);
  return node;
}

GCXX_FH auto GraphView::AddMemcpyNode(
  const deviceGraphNode_t* pDependencies, size_t numDependencies,
  const details_::deviceMemcpy3DParms_t* copyParams) -> deviceGraphNode_t {
  deviceGraphNode_t node;
  GCXX_SAFE_RUNTIME_CALL(GraphAddMemcpyNode,
                         "Failed to add memcpy node to graph", &node, graph_,
                         pDependencies, numDependencies, copyParams);
  return node;
}

GCXX_FH auto GraphView::AddMemcpyNode1D(const deviceGraphNode_t* pDependencies,
                                        size_t numDependencies, void* dst,
                                        const void* src, size_t count,
                                        deviceMemcpyKind kind)
  -> deviceGraphNode_t {
  deviceGraphNode_t node;
  GCXX_SAFE_RUNTIME_CALL(GraphAddMemcpyNode1D,
                         "Failed to add 1D memcpy node to graph", &node, graph_,
                         pDependencies, numDependencies, dst, src, count, kind);
  return node;
}

GCXX_FH auto GraphView::AddMemcpyNodeFromSymbol(
  const deviceGraphNode_t* pDependencies, size_t numDependencies, void* dst,
  const void* symbol, size_t count, size_t offset, deviceMemcpyKind kind)
  -> deviceGraphNode_t {
  deviceGraphNode_t node;
  GCXX_SAFE_RUNTIME_CALL(GraphAddMemcpyNodeFromSymbol,
                         "Failed to add memcpy from symbol node to graph",
                         &node, graph_, pDependencies, numDependencies, dst,
                         symbol, count, offset, kind);
  return node;
}

GCXX_FH auto GraphView::AddMemcpyNodeToSymbol(
  const deviceGraphNode_t* pDependencies, size_t numDependencies,
  const void* symbol, const void* src, size_t count, size_t offset,
  deviceMemcpyKind kind) -> deviceGraphNode_t {
  deviceGraphNode_t node;
  GCXX_SAFE_RUNTIME_CALL(GraphAddMemcpyNodeToSymbol,
                         "Failed to add memcpy to symbol node to graph", &node,
                         graph_, pDependencies, numDependencies, symbol, src,
                         count, offset, kind);
  return node;
}

GCXX_FH auto GraphView::AddMemsetNode(
  const deviceGraphNode_t* pDependencies, size_t numDependencies,
  const details_::deviceMemsetParams_t* memsetParams) -> deviceGraphNode_t {
  deviceGraphNode_t node;
  GCXX_SAFE_RUNTIME_CALL(GraphAddMemsetNode,
                         "Failed to add memset node to graph", &node, graph_,
                         pDependencies, numDependencies, memsetParams);
  return node;
}

GCXX_FH auto GraphView::AddNode(const deviceGraphNode_t* pDependencies,
                                size_t numDependencies,
                                details_::deviceGraphNodeParams_t* nodeParams)
  -> deviceGraphNode_t {
  deviceGraphNode_t node;
  GCXX_SAFE_RUNTIME_CALL(GraphAddNode, "Failed to add node to graph", &node,
                         graph_, pDependencies, numDependencies, nodeParams);
  return node;
}

GCXX_NAMESPACE_MAIN_END

#include <gcxx/macros/undefine_macros.hpp>

#endif
