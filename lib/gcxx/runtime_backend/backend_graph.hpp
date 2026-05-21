// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_GRAPH_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_GRAPH_HPP_

#include <cstddef>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime_backend/backend_handles.hpp>

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN


#if GCXX_CUDA_MODE()
GCXX_FH auto deviceGetGraphMemAttribute(int device,
                                        deviceGraphMemAttributeType_t attr,
                                        void* value) -> void {
  GCXX_SAFE_RUNTIME_CALL(DeviceGetGraphMemAttribute,
                         "Failed to get device graph memory attribute", device,
                         attr, value);
}

GCXX_FH auto deviceSetGraphMemAttribute(int device,
                                        deviceGraphMemAttributeType_t attr,
                                        void* value) -> void {
  GCXX_SAFE_RUNTIME_CALL(DeviceSetGraphMemAttribute,
                         "Failed to set device graph memory attribute", device,
                         attr, value);
}

GCXX_FH auto deviceGraphMemTrim(int device) -> void {
  GCXX_SAFE_RUNTIME_CALL(DeviceGraphMemTrim,
                         "Failed to trim device graph memory", device);
}

GCXX_FD auto getCurrentGraphExec() -> deviceGraphExec_t {
  return GCXX_RUNTIME_BACKEND(GetCurrentGraphExec)();
}
#endif
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
#if GCXX_CUDA_MODE()
GCXX_FH auto graphConditionalHandleCreate(
  deviceGraph_t graph, unsigned int defaultLaunchValue,
  details_::flag_t flag) -> deviceGraphConditionalHandle_t {
  deviceGraphConditionalHandle_t out{0};
  GCXX_SAFE_RUNTIME_CALL(GraphConditionalHandleCreate,
                         "Failed to create conditional handle in graph", &out,
                         graph, defaultLaunchValue, flag);
  return out;
}

GCXX_FD
auto graphSetConditional(deviceGraphConditionalHandle_t handle,
                         unsigned int value) -> void {
  GCXX_RUNTIME_BACKEND(GraphSetConditional)(handle, value);
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
                                  const deviceGraphEdgeData_t* edgeData,
                                  std::size_t numDependencies) -> void {
  GCXX_SAFE_RUNTIME_CALL(
#if GCXX_CUDA_VERSION_LESS_THAN(13, 0, 0)
    GraphAddDependencies_v2,
#else
    GraphAddDependencies,
#endif
    "Failed to Add Dependency between graph Nodes", graph, from, to,
#if GCXX_CUDA_MODE()
    edgeData,
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
#if GCXX_CUDA_MODE()
GCXX_FH auto graphAddExternalSemaphoresSignalNode(
  deviceGraph_t graph, const deviceExternalSemaphoreSignalNodeParams_t* params,
  const deviceGraphNode_t* dependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  deviceGraphNode_t node{INVALID_GRAPH_NODE};
  GCXX_SAFE_RUNTIME_CALL(
    GraphAddExternalSemaphoresSignalNode,
    "Failed to add external semaphore signal node to graph", &node, graph,
    dependencies, numDependencies, params);
  return node;
}

GCXX_FH auto graphAddExternalSemaphoresWaitNode(
  deviceGraph_t graph, const deviceExternalSemaphoreWaitNodeParams_t* params,
  const deviceGraphNode_t* dependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  deviceGraphNode_t node{INVALID_GRAPH_NODE};
  GCXX_SAFE_RUNTIME_CALL(GraphAddExternalSemaphoresWaitNode,
                         "Failed to add external semaphore wait node to graph",
                         &node, graph, dependencies, numDependencies, params);
  return node;
}

GCXX_FH auto graphAddMemAllocNode(
  deviceGraph_t graph, deviceMemAllocNodeParams_t* params,
  const deviceGraphNode_t* dependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  deviceGraphNode_t node{INVALID_GRAPH_NODE};
  GCXX_SAFE_RUNTIME_CALL(GraphAddMemAllocNode,
                         "Failed to Add Mem alloc Node to Graph", &node, graph,
                         dependencies, numDependencies, params);
  return node;
}
#endif

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
#if GCXX_CUDA_MODE()
GCXX_FH auto graphAddMemcpyNodeFromSymbol(
  deviceGraph_t graph, void* dst, const void* symbol, std::size_t count,
  std::size_t offset, deviceMemcpyKind_t kind,
  const deviceGraphNode_t* dependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  deviceGraphNode_t node{INVALID_GRAPH_NODE};
  GCXX_SAFE_RUNTIME_CALL(GraphAddMemcpyNodeFromSymbol,
                         "Failed to Add Memcpy From Symbol Node to Graph",
                         &node, graph, dependencies, numDependencies, dst,
                         symbol, count, offset, kind);
  return node;
}

GCXX_FH auto graphAddMemcpyNodeToSymbol(
  deviceGraph_t graph, const void* symbol, const void* src, std::size_t count,
  std::size_t offset, deviceMemcpyKind_t kind,
  const deviceGraphNode_t* dependencies,
  std::size_t numDependencies) -> deviceGraphNode_t {
  deviceGraphNode_t node{INVALID_GRAPH_NODE};
  GCXX_SAFE_RUNTIME_CALL(GraphAddMemcpyNodeToSymbol,
                         "Failed to Add Memcpy To Symbol Node to Graph", &node,
                         graph, dependencies, numDependencies, symbol, src,
                         count, offset, kind);
  return node;
}
#endif

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
#if GCXX_CUDA_MODE()
GCXX_FH auto graphAddNode(deviceGraphNode_t* pGraphNode, deviceGraph_t graph,
                          const deviceGraphNode_t* pDependencies,
                          const deviceGraphEdgeData_t* dependencyData,
                          size_t numDependencies,
                          deviceGraphNodeParams_t* nodeParams) -> void {
  GCXX_SAFE_RUNTIME_CALL(
#if GCXX_CUDA_VERSION_LESS_THAN(13, 0, 0)
    GraphAddNode_v2,
#else
    GraphAddNode,
#endif
    "Failed to add node to graph", pGraphNode, graph, pDependencies,
#if GCXX_CUDA_MODE()
    dependencyData,
#endif
    numDependencies, nodeParams);
}
#endif

#if GCXX_CUDA_MODE()
GCXX_FH auto graphDestroyNode(deviceGraphNode_t node) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphDestroyNode, "Failed to destroy graph node",
                         node);
}

GCXX_FH auto graphEventRecordNodeGetEvent(deviceGraphNode_t node)
  -> deviceEvent_t {
  deviceEvent_t event{INVALID_EVENT};
  GCXX_SAFE_RUNTIME_CALL(GraphEventRecordNodeGetEvent,
                         "Failed to get event record node event", node, &event);
  return event;
}

GCXX_FH auto graphEventRecordNodeSetEvent(deviceGraphNode_t node,
                                          deviceEvent_t event) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphEventRecordNodeSetEvent,
                         "Failed to set event record node event", node, event);
}

GCXX_FH auto graphEventWaitNodeGetEvent(deviceGraphNode_t node)
  -> deviceEvent_t {
  deviceEvent_t event{INVALID_EVENT};
  GCXX_SAFE_RUNTIME_CALL(GraphEventWaitNodeGetEvent,
                         "Failed to get event wait node event", node, &event);
  return event;
}

GCXX_FH auto graphEventWaitNodeSetEvent(deviceGraphNode_t node,
                                        deviceEvent_t event) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphEventWaitNodeSetEvent,
                         "Failed to set event wait node event", node, event);
}
#if GCXX_CUDA_MODE()
GCXX_FH auto graphExternalSemaphoresSignalNodeGetParams(
  deviceGraphNode_t node,
  deviceExternalSemaphoreSignalNodeParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphExternalSemaphoresSignalNodeGetParams,
                         "Failed to get external semaphore signal node params",
                         node, params);
}

GCXX_FH auto graphExternalSemaphoresSignalNodeSetParams(
  deviceGraphNode_t node,
  const deviceExternalSemaphoreSignalNodeParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphExternalSemaphoresSignalNodeSetParams,
                         "Failed to set external semaphore signal node params",
                         node, params);
}

GCXX_FH auto graphExternalSemaphoresWaitNodeGetParams(
  deviceGraphNode_t node,
  deviceExternalSemaphoreWaitNodeParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphExternalSemaphoresWaitNodeGetParams,
                         "Failed to get external semaphore wait node params",
                         node, params);
}

GCXX_FH auto graphExternalSemaphoresWaitNodeSetParams(
  deviceGraphNode_t node,
  const deviceExternalSemaphoreWaitNodeParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphExternalSemaphoresWaitNodeSetParams,
                         "Failed to set external semaphore wait node params",
                         node, params);
}
#endif

GCXX_FH auto graphGetNodes(deviceGraph_t graph, deviceGraphNode_t* nodes,
                           std::size_t* numNodes) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphGetNodes, "Failed to get graph nodes", graph,
                         nodes, numNodes);
}

GCXX_FH auto graphGetEdges(deviceGraph_t graph, deviceGraphNode_t* from,
                           deviceGraphNode_t* to,
                           deviceGraphEdgeData_t* edgeData,
                           std::size_t* numEdges) -> void {
  GCXX_SAFE_RUNTIME_CALL(
#if GCXX_CUDA_VERSION_LESS_THAN(13, 0, 0)
    GraphGetEdges_v2,
#else
    GraphGetEdges,
#endif
    "Failed to get graph edges", graph, from, to,
#if GCXX_CUDA_MODE()
    edgeData,
#endif
    numEdges);
}

GCXX_FH auto graphGetRootNodes(deviceGraph_t graph,
                               deviceGraphNode_t* rootNodes,
                               std::size_t* numRootNodes) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphGetRootNodes, "Failed to get graph root nodes",
                         graph, rootNodes, numRootNodes);
}

GCXX_FH auto graphHostNodeGetParams(deviceGraphNode_t node,
                                    deviceHostNodeParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphHostNodeGetParams,
                         "Failed to get host node params", node, params);
}

GCXX_FH auto graphHostNodeSetParams(
  deviceGraphNode_t node, const deviceHostNodeParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphHostNodeSetParams,
                         "Failed to set host node params", node, params);
}
#endif

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
#if GCXX_CUDA_MODE()
GCXX_FH auto graphExecEventRecordNodeSetEvent(deviceGraphExec_t exec,
                                              deviceGraphNode_t node,
                                              deviceEvent_t event) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphExecEventRecordNodeSetEvent,
                         "Failed to set graph exec event record node event",
                         exec, node, event);
}

GCXX_FH auto graphExecEventWaitNodeSetEvent(deviceGraphExec_t exec,
                                            deviceGraphNode_t node,
                                            deviceEvent_t event) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphExecEventWaitNodeSetEvent,
                         "Failed to set graph exec event wait node event", exec,
                         node, event);
}
#if GCXX_CUDA_MODE()
GCXX_FH auto graphExecExternalSemaphoresSignalNodeSetParams(
  deviceGraphExec_t exec, deviceGraphNode_t node,
  const deviceExternalSemaphoreSignalNodeParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(
    GraphExecExternalSemaphoresSignalNodeSetParams,
    "Failed to set graph exec external semaphore signal node params", exec,
    node, params);
}

GCXX_FH auto graphExecExternalSemaphoresWaitNodeSetParams(
  deviceGraphExec_t exec, deviceGraphNode_t node,
  const deviceExternalSemaphoreWaitNodeParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(
    GraphExecExternalSemaphoresWaitNodeSetParams,
    "Failed to set graph exec external semaphore wait node params", exec, node,
    params);
}
#endif
GCXX_FH auto graphExecGetFlags(deviceGraphExec_t exec) -> unsigned long long {
  unsigned long long flags{};
  GCXX_SAFE_RUNTIME_CALL(GraphExecGetFlags, "Failed to get graph exec flags",
                         exec, &flags);
  return flags;
}

GCXX_FH auto graphExecHostNodeSetParams(
  deviceGraphExec_t exec, deviceGraphNode_t node,
  const deviceHostNodeParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphExecHostNodeSetParams,
                         "Failed to set graph exec host node params", exec,
                         node, params);
}

GCXX_FH auto graphExecKernelNodeSetParams(
  deviceGraphExec_t exec, deviceGraphNode_t node,
  const deviceKernelNodeParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphExecKernelNodeSetParams,
                         "Failed to set graph exec kernel node params", exec,
                         node, params);
}

GCXX_FH auto graphExecMemcpyNodeSetParams(
  deviceGraphExec_t exec, deviceGraphNode_t node,
  const deviceMemcpy3DParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphExecMemcpyNodeSetParams,
                         "Failed to set graph exec memcpy node params", exec,
                         node, params);
}

GCXX_FH auto graphExecMemcpyNodeSetParams1D(deviceGraphExec_t exec,
                                            deviceGraphNode_t node, void* dst,
                                            const void* src, std::size_t count,
                                            deviceMemcpyKind_t kind) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphExecMemcpyNodeSetParams1D,
                         "Failed to set graph exec memcpy1D node params", exec,
                         node, dst, src, count, kind);
}
#if GCXX_CUDA_MODE()
GCXX_FH auto graphExecMemcpyNodeSetParamsFromSymbol(
  deviceGraphExec_t exec, deviceGraphNode_t node, void* dst, const void* symbol,
  std::size_t count, std::size_t offset, deviceMemcpyKind_t kind) -> void {
  GCXX_SAFE_RUNTIME_CALL(
    GraphExecMemcpyNodeSetParamsFromSymbol,
    "Failed to set graph exec memcpy from symbol node params", exec, node, dst,
    symbol, count, offset, kind);
}

GCXX_FH auto graphExecMemcpyNodeSetParamsToSymbol(
  deviceGraphExec_t exec, deviceGraphNode_t node, const void* symbol,
  const void* src, std::size_t count, std::size_t offset,
  deviceMemcpyKind_t kind) -> void {
  GCXX_SAFE_RUNTIME_CALL(
    GraphExecMemcpyNodeSetParamsToSymbol,
    "Failed to set graph exec memcpy to symbol node params", exec, node, symbol,
    src, count, offset, kind);
}
#endif

GCXX_FH auto graphExecMemsetNodeSetParams(
  deviceGraphExec_t exec, deviceGraphNode_t node,
  const deviceMemsetParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphExecMemsetNodeSetParams,
                         "Failed to set graph exec memset node params", exec,
                         node, params);
}
#if GCXX_CUDA_MODE()
GCXX_FH auto graphExecNodeSetParams(deviceGraphExec_t exec,
                                    deviceGraphNode_t node,
                                    deviceGraphNodeParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphExecNodeSetParams,
                         "Failed to set graph exec node params", exec, node,
                         params);
}
#endif

#endif
GCXX_FH auto graphInstantiate(deviceGraph_t graph) -> deviceGraphExec_t {
  deviceGraphExec_t exec{INVALID_GRAPH_EXEC};
  GCXX_SAFE_RUNTIME_CALL(GraphInstantiate, "Failed to instantiate the graph",
                         &exec, graph, nullptr, nullptr, 0);
  return exec;
}
#if GCXX_CUDA_MODE()
GCXX_FH auto graphInstantiateWithFlags(
  deviceGraph_t graph, unsigned long long flags) -> deviceGraphExec_t {
  deviceGraphExec_t exec{INVALID_GRAPH_EXEC};
  GCXX_SAFE_RUNTIME_CALL(GraphInstantiateWithFlags,
                         "Failed to instantiate the graph with flags", &exec,
                         graph, flags);
  return exec;
}

GCXX_FH auto graphInstantiateWithParams(deviceGraph_t graph,
                                        deviceGraphInstantiateParams_t* params)
  -> deviceGraphExec_t {
  deviceGraphExec_t exec{INVALID_GRAPH_EXEC};
  GCXX_SAFE_RUNTIME_CALL(GraphInstantiateWithParams,
                         "Failed to instantiate the graph with params", &exec,
                         graph, params);
  return exec;
}
#endif
GCXX_FH auto graphExecDestroy(deviceGraphExec_t exec) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphExecDestroy, "Failed to destroy graph exec",
                         exec);
}
#if GCXX_CUDA_MODE()
GCXX_FH auto graphExecUpdate(deviceGraphExec_t exec, deviceGraph_t graph,
                             deviceGraphExecUpdateResultInfo_t* resultInfo)
  -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphExecUpdate, "Failed to update graph exec", exec,
                         graph, resultInfo);
}
#endif

GCXX_FH auto graphExecUpdate(deviceGraphExec_t exec,
                             deviceGraph_t graph) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphExecUpdate, "Failed to update graph exec", exec,
                         graph,
#if GCXX_HIP_MODE()
                         NULL,
#endif
                         nullptr);
}
#if GCXX_CUDA_MODE()
GCXX_FH auto graphKernelNodeCopyAttributes(deviceGraphNode_t src,
                                           deviceGraphNode_t dst) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphKernelNodeCopyAttributes,
                         "Failed to copy kernel node attributes", src, dst);
}

GCXX_FH auto graphKernelNodeGetAttribute(
  deviceGraphNode_t node, deviceKernelNodeAttrID_t attr,
  deviceKernelNodeAttrValue_t* value) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphKernelNodeGetAttribute,
                         "Failed to get kernel node attribute", node, attr,
                         value);
}

GCXX_FH auto graphKernelNodeGetParams(
  deviceGraphNode_t node, deviceKernelNodeParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphKernelNodeGetParams,
                         "Failed to get kernel node parameters", node, params);
}

GCXX_FH auto graphKernelNodeSetAttribute(
  deviceGraphNode_t node, deviceKernelNodeAttrID_t attr,
  const deviceKernelNodeAttrValue_t* value) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphKernelNodeSetAttribute,
                         "Failed to set kernel node attribute", node, attr,
                         value);
}

GCXX_FH auto graphKernelNodeSetParams(
  deviceGraphNode_t node, const deviceKernelNodeParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphKernelNodeSetParams,
                         "Failed to set kernel node parameters", node, params);
}
#if GCXX_CUDA_MODE()
GCXX_FH auto graphMemAllocNodeGetParams(
  deviceGraphNode_t node, deviceMemAllocNodeParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphMemAllocNodeGetParams,
                         "Failed to get mem alloc node params", node, params);
}
#endif

GCXX_FH auto graphMemFreeNodeGetParams(deviceGraphNode_t node,
                                       void* dptrOut) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphMemFreeNodeGetParams,
                         "Failed to get mem free node params", node, dptrOut);
}

GCXX_FH auto graphMemcpyNodeGetParams(deviceGraphNode_t node,
                                      deviceMemcpy3DParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphMemcpyNodeGetParams,
                         "Failed to get memcpy node params", node, params);
}

GCXX_FH auto graphMemcpyNodeSetParams(
  deviceGraphNode_t node, const deviceMemcpy3DParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphMemcpyNodeSetParams,
                         "Failed to set memcpy node params", node, params);
}

GCXX_FH auto graphMemcpyNodeSetParams1D(deviceGraphNode_t node, void* dst,
                                        const void* src, std::size_t count,
                                        deviceMemcpyKind_t kind) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphMemcpyNodeSetParams1D,
                         "Failed to set memcpy1D node params", node, dst, src,
                         count, kind);
}
#if GCXX_CUDA_MODE()
GCXX_FH auto graphMemcpyNodeSetParamsFromSymbol(
  deviceGraphNode_t node, void* dst, const void* symbol, std::size_t count,
  std::size_t offset, deviceMemcpyKind_t kind) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphMemcpyNodeSetParamsFromSymbol,
                         "Failed to set memcpy from symbol node params", node,
                         dst, symbol, count, offset, kind);
}

GCXX_FH auto graphMemcpyNodeSetParamsToSymbol(
  deviceGraphNode_t node, const void* symbol, const void* src,
  std::size_t count, std::size_t offset, deviceMemcpyKind_t kind) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphMemcpyNodeSetParamsToSymbol,
                         "Failed to set memcpy to symbol node params", node,
                         symbol, src, count, offset, kind);
}
#endif

GCXX_FH auto graphMemsetNodeGetParams(deviceGraphNode_t node,
                                      deviceMemsetParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphMemsetNodeGetParams,
                         "Failed to get memset node params", node, params);
}

GCXX_FH auto graphMemsetNodeSetParams(
  deviceGraphNode_t node, const deviceMemsetParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphMemsetNodeSetParams,
                         "Failed to set memset node params", node, params);
}

GCXX_FH auto graphNodeFindInClone(deviceGraphNode_t originalNode,
                                  deviceGraph_t clonedGraph)
  -> deviceGraphNode_t {
  deviceGraphNode_t node{INVALID_GRAPH_NODE};
  GCXX_SAFE_RUNTIME_CALL(GraphNodeFindInClone,
                         "Failed to find graph node in clone", &node,
                         originalNode, clonedGraph);
  return node;
}

GCXX_FH auto graphNodeGetDependencies(deviceGraphNode_t node,
                                      deviceGraphNode_t* dependencies,
                                      deviceGraphEdgeData_t* edgeData,
                                      std::size_t* numDependencies) -> void {
  GCXX_SAFE_RUNTIME_CALL(
#if GCXX_CUDA_VERSION_LESS_THAN(13, 0, 0)
    GraphNodeGetDependencies_v2,
#else
    GraphNodeGetDependencies,
#endif
    "Failed to get graph node dependencies", node, dependencies,
#if GCXX_CUDA_MODE()
    edgeData,
#endif
    numDependencies);
}

GCXX_FH auto graphNodeGetDependentNodes(deviceGraphNode_t node,
                                        deviceGraphNode_t* pDependentNodes,
                                        deviceGraphEdgeData_t* edgeData,
                                        size_t* pNumDependentNodes) -> void {
  GCXX_SAFE_RUNTIME_CALL(
#if GCXX_CUDA_VERSION_LESS_THAN(13, 0, 0)
    GraphNodeGetDependentNodes_v2,
#else
    GraphNodeGetDependentNodes,
#endif
    "Failed to get graph node dependents", node, pDependentNodes,
#if GCXX_CUDA_MODE()
    edgeData,
#endif
    pNumDependentNodes);
}

GCXX_FH auto graphNodeGetEnabled(deviceGraphExec_t exec,
                                 deviceGraphNode_t node) -> unsigned int {
  unsigned int isEnabled{};
  GCXX_SAFE_RUNTIME_CALL(GraphNodeGetEnabled,
                         "Failed to get graph node enabled state", exec, node,
                         &isEnabled);
  return isEnabled;
}

GCXX_FH auto graphNodeSetEnabled(deviceGraphExec_t exec, deviceGraphNode_t node,
                                 unsigned int isEnabled) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphNodeSetEnabled,
                         "Failed to set graph node enabled state", exec, node,
                         isEnabled);
}
#if GCXX_CUDA_MODE()
GCXX_FH auto graphNodeSetParams(deviceGraphNode_t node,
                                deviceGraphNodeParams_t* params) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphNodeSetParams, "Failed to set graph node params",
                         node, params);
}
#endif

#if GCXX_CUDA_MODE()
GCXX_FH auto graphRetainUserObject(deviceGraph_t graph,
                                   deviceUserObject_t object,
                                   unsigned int count = 1,
                                   unsigned int flags = 0) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphRetainUserObject,
                         "Failed to retain graph user object", graph, object,
                         count, flags);
}

GCXX_FH auto graphReleaseUserObject(deviceGraph_t graph,
                                    deviceUserObject_t object,
                                    unsigned int count = 1) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphReleaseUserObject,
                         "Failed to release graph user object", graph, object,
                         count);
}
#endif

GCXX_FH auto graphRemoveDependencies(deviceGraph_t graph,
                                     const deviceGraphNode_t* from,
                                     const deviceGraphNode_t* to,
                                     std::size_t numDependencies) -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphRemoveDependencies,
                         "Failed to remove graph dependencies", graph, from, to,
#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)
                         nullptr,
#endif
                         numDependencies);
}

#if GCXX_CUDA_MODE()
GCXX_FH auto userObjectCreate(deviceUserObject_t* objectOut, void* ptr,
                              deviceHostCallBackFn_t destroy,
                              unsigned int initialRefcount,
                              unsigned int flags) -> void {
  GCXX_SAFE_RUNTIME_CALL(UserObjectCreate, "Failed to create user object",
                         objectOut, ptr, destroy, initialRefcount, flags);
}

GCXX_FH auto userObjectRelease(deviceUserObject_t object,
                               unsigned int count = 1) -> void {
  GCXX_SAFE_RUNTIME_CALL(UserObjectRelease, "Failed to release user object",
                         object, count);
}

GCXX_FH auto userObjectRetain(deviceUserObject_t object,
                              unsigned int count = 1) -> void {
  GCXX_SAFE_RUNTIME_CALL(UserObjectRetain, "Failed to retain user object",
                         object, count);
}
#endif

#endif

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
