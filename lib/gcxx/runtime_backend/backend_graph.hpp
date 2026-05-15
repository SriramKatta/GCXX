// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_GRAPH_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_GRAPH_HPP_

#include <cstddef>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime_backend/backend_handles.hpp>

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN

GCXX_FH auto graphCreate(details_::flag_t createFlag) -> deviceGraph_t {
  deviceGraph_t graph{INVALID_GRAPH};
  GCXX_SAFE_RUNTIME_CALL(GraphCreate, "Failed to create the graph", &graph,
                         createFlag);
  return graph;
}

GCXX_FH auto graphDestroy(deviceGraph_t graph) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphDestroy, "Failed to destroy the graph", graph);
}

GCXX_FH auto graphDebugDotPrint(deviceGraph_t graph, const char* filename,
                                details_::flag_t flag) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphDebugDotPrint,
                         "Failed to output the dot file of the graph", graph,
                         filename, flag);
}

GCXX_FH auto graphGetNumNodes(deviceGraph_t graph) -> std::size_t {
  std::size_t numNodes{};
  GCXX_SAFE_RUNTIME_CALL(GraphGetNodes, "Failed to get Count of Graph nodes",
                         graph, nullptr, &numNodes);
  return numNodes;
}

GCXX_FH auto graphGetNumEdges(deviceGraph_t graph) -> std::size_t {
  std::size_t numEdges{};
  GCXX_SAFE_RUNTIME_CALL(GraphGetEdges, "Failed to get count of Graph edges",
                         graph, nullptr, nullptr,
#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)
                         nullptr,
#endif
                         &numEdges);
  return numEdges;
}

GCXX_FH auto graphClone(deviceGraph_t graph) -> deviceGraph_t {
  deviceGraph_t clonedGraph{INVALID_GRAPH};
  GCXX_SAFE_RUNTIME_CALL(GraphClone, "Failed to clone graph", &clonedGraph,
                         graph);
  return clonedGraph;
}

#if GCXX_CUDA_MODE
GCXX_FH auto graphConditionalHandleCreate(
  deviceGraph_t graph, unsigned int defaultLaunchValue,
  details_::flag_t flag) -> deviceGraphConditionalHandle_t {
  deviceGraphConditionalHandle_t out{0};
  GCXX_SAFE_RUNTIME_CALL(GraphConditionalHandleCreate,
                         "Failed to create conditional handle in graph", &out,
                         graph, defaultLaunchValue, flag);
  return out;
}

GCXX_FD auto graphSetConditional(deviceGraphConditionalHandle_t handle,
                                 unsigned int value) -> void {
  GCXX_RUNTIME_BACKEND(GraphSetConditional)(handle, value);
}

GCXX_FH auto graphAddNode(
  deviceGraph_t graph, const deviceGraphNode_t* dependencies,
  std::size_t numDependencies,
  deviceGraphNodeParams_t* params) -> deviceGraphNode_t {
  deviceGraphNode_t node{INVALID_GRAPH_NODE};
  GCXX_SAFE_RUNTIME_CALL(GraphAddNode,
                         "Failed to add conditional node to graph", &node,
                         graph, dependencies,
#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)
                         nullptr,
#endif
                         numDependencies, params);
  return node;
}
#endif

GCXX_FH auto graphAddChildGraphNode(
  deviceGraph_t graph, deviceGraph_t childGraph,
  const deviceGraphNode_t* dependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  deviceGraphNode_t node{INVALID_GRAPH_NODE};
  GCXX_SAFE_RUNTIME_CALL(GraphAddChildGraphNode,
                         "Failed to Add Child graph Node to Graph", &node,
                         graph, dependencies, numDependencies, childGraph);
  return node;
}

GCXX_FH auto graphAddDependencies(deviceGraph_t graph,
                                  const deviceGraphNode_t* from,
                                  const deviceGraphNode_t* to,
                                  std::size_t numDependencies) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphAddDependencies,
                         "Failed to Add Dependency between graph Nodes", graph,
                         from, to,
#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)
                         nullptr,
#endif
                         numDependencies);
}

GCXX_FH auto graphAddEmptyNode(
  deviceGraph_t graph, const deviceGraphNode_t* dependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  deviceGraphNode_t node{INVALID_GRAPH_NODE};
  GCXX_SAFE_RUNTIME_CALL(GraphAddEmptyNode, "Failed to Add Empty Node to Graph",
                         &node, graph, dependencies, numDependencies);
  return node;
}

GCXX_FH auto graphAddEventRecordNode(deviceGraph_t graph, deviceEvent_t event,
                                     const deviceGraphNode_t* dependencies,
                                     std::size_t numDependencies)
  -> deviceGraphNode_t {
  deviceGraphNode_t node{INVALID_GRAPH_NODE};
  GCXX_SAFE_RUNTIME_CALL(GraphAddEventRecordNode,
                         "Failed to Add Event record Node to Graph", &node,
                         graph, dependencies, numDependencies, event);
  return node;
}

GCXX_FH auto graphAddEventWaitNode(deviceGraph_t graph, deviceEvent_t event,
                                   const deviceGraphNode_t* dependencies,
                                   std::size_t numDependencies)
  -> deviceGraphNode_t {
  deviceGraphNode_t node{INVALID_GRAPH_NODE};
  GCXX_SAFE_RUNTIME_CALL(GraphAddEventWaitNode,
                         "Failed to Add Event Wait Node to Graph", &node, graph,
                         dependencies, numDependencies, event);
  return node;
}

GCXX_FH auto graphAddHostNode(
  deviceGraph_t graph, const deviceHostNodeParams_t* params,
  const deviceGraphNode_t* dependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  deviceGraphNode_t node{INVALID_GRAPH_NODE};
  GCXX_SAFE_RUNTIME_CALL(GraphAddHostNode, "Failed to Add Host Node to Graph",
                         &node, graph, dependencies, numDependencies, params);
  return node;
}

GCXX_FH auto graphAddKernelNode(
  deviceGraph_t graph, const deviceKernelNodeParams_t* params,
  const deviceGraphNode_t* dependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  deviceGraphNode_t node{INVALID_GRAPH_NODE};
  GCXX_SAFE_RUNTIME_CALL(GraphAddKernelNode,
                         "Failed to Add Kernel Node to Graph", &node, graph,
                         dependencies, numDependencies, params);
  return node;
}

GCXX_FH auto graphAddMemFreeNode(
  deviceGraph_t graph, void* dptr, const deviceGraphNode_t* dependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  deviceGraphNode_t node{INVALID_GRAPH_NODE};
  GCXX_SAFE_RUNTIME_CALL(GraphAddMemFreeNode,
                         "Failed to Add Mem free Node to Graph", &node, graph,
                         dependencies, numDependencies, dptr);
  return node;
}

GCXX_FH auto graphAddMemcpyNode(
  deviceGraph_t graph, const deviceMemcpy3DParams_t* params,
  const deviceGraphNode_t* dependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  deviceGraphNode_t node{INVALID_GRAPH_NODE};
  GCXX_SAFE_RUNTIME_CALL(GraphAddMemcpyNode,
                         "Failed to Add Memcpy Node to Graph", &node, graph,
                         dependencies, numDependencies, params);
  return node;
}

GCXX_FH auto graphAddMemcpyNode1D(
  deviceGraph_t graph, void* dst, const void* src, std::size_t countBytes,
  const deviceGraphNode_t* dependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  deviceGraphNode_t node{INVALID_GRAPH_NODE};
  GCXX_SAFE_RUNTIME_CALL(GraphAddMemcpyNode1D,
                         "Failed to Add Memcpy1D Node to Graph", &node, graph,
                         dependencies, numDependencies, dst, src, countBytes,
                         GCXX_RUNTIME_BACKEND(MemcpyDefault));
  return node;
}

GCXX_FH auto graphAddMemsetNode(
  deviceGraph_t graph, const deviceMemsetParams_t* params,
  const deviceGraphNode_t* dependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  deviceGraphNode_t node{INVALID_GRAPH_NODE};
  GCXX_SAFE_RUNTIME_CALL(GraphAddMemsetNode,
                         "Failed to Add Memset Node to Graph", &node, graph,
                         dependencies, numDependencies, params);
  return node;
}

GCXX_FH auto graphNodeGetType(deviceGraphNode_t node) -> deviceGraphNodeType_t {
  deviceGraphNodeType_t enumval{};
  GCXX_SAFE_RUNTIME_CALL(GraphNodeGetType,
                         "Failed to query the graph node type", node, &enumval);
  return enumval;
}

#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 1, 0)
GCXX_FH auto graphNodeGetLocalId(deviceGraphNode_t node) -> unsigned int {
  unsigned int id{};
  GCXX_SAFE_RUNTIME_CALL(GraphNodeGetLocalId,
                         "Failed to query Local Id of graph node", node, &id);
  return id;
}

GCXX_FH auto graphNodeGetToolsId(deviceGraphNode_t node) -> unsigned long long {
  unsigned long long id{};
  GCXX_SAFE_RUNTIME_CALL(GraphNodeGetToolsId,
                         "Failed to query Tools Id of graph node", node, &id);
  return id;
}
#endif

GCXX_FH auto graphChildGraphNodeGetGraph(deviceGraphNode_t node)
  -> deviceGraph_t {
  deviceGraph_t graph{INVALID_GRAPH};
  GCXX_SAFE_RUNTIME_CALL(GraphChildGraphNodeGetGraph,
                         "Failed to get the graph of given Node", node, &graph);
  return graph;
}

GCXX_FH auto graphExecChildGraphNodeSetParams(deviceGraphExec_t exec,
                                              deviceGraphNode_t node,
                                              deviceGraph_t graph) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphExecChildGraphNodeSetParams,
                         "Failed to set child graph for Graph exex", exec, node,
                         graph);
}

GCXX_FH auto graphInstantiate(deviceGraph_t graph) -> deviceGraphExec_t {
  deviceGraphExec_t exec{INVALID_GRAPH_EXEC};
  GCXX_SAFE_RUNTIME_CALL(GraphInstantiate, "Failed to instantiate the graph",
                         &exec, graph, nullptr, nullptr, 0);
  return exec;
}

GCXX_FH auto graphExecDestroy(deviceGraphExec_t exec) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphExecDestroy, "Failed to destroy graph exec",
                         exec);
}

GCXX_FH auto graphExecUpdate(deviceGraphExec_t exec,
                             deviceGraph_t graph) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphExecUpdate, "Failed to update graph exec", exec,
                         graph,
#if GCXX_HIP_MODE
                         NULL,
#endif
                         nullptr);
}

GCXX_FH auto graphLaunch(deviceGraphExec_t exec,
                         deviceStream_t stream) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphLaunch, "Failed to launch graph", exec, stream);
}

GCXX_FH auto graphUpload(deviceGraphExec_t exec,
                         deviceStream_t stream) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphUpload, "Failed to upload graph", exec, stream);
}

GCXX_NAMESPACE_MAIN_DRIVER_END

#endif
