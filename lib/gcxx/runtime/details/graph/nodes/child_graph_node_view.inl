#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_NODES_CHILD_GRAPH_NODE_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_NODES_CHILD_GRAPH_NODE_VIEW_INL_

#include <gcxx/internal/prologue.hpp>


#include <gcxx/runtime/graph/graph_exec_view.hpp>
#include <gcxx/runtime/graph/graph_view.hpp>


GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FHC ChildGraphNodeView::ChildGraphNodeView(deviceGraphNode_t node)
    : GraphNodeView(node) {}

GCXX_FH auto ChildGraphNodeView::getGraph() -> GraphView {
  deviceGraph_t graph = nullptr;
  GCXX_SAFE_RUNTIME_CALL(GraphChildGraphNodeGetGraph,
                         "Failed to get the graph of given Node", node_,
                         &graph);
  return {graph};
}

GCXX_FH auto ChildGraphNodeView::setParams(GraphExecView exec, GraphView graph)
  -> void {
  GCXX_SAFE_RUNTIME_CALL(GraphExecChildGraphNodeSetParams,
                         "Failed to set child graph for Graph exex",
                         exec.getRawExec(), node_, graph.getRawGraph());
}

GCXX_NAMESPACE_MAIN_END

#endif