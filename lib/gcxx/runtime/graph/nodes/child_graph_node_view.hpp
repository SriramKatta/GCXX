#pragma once
#ifndef GCXX_RUNTIME_GRAPH_NODES_CHILD_GRAPH_NODE_VIEW_HPP_
#define GCXX_RUNTIME_GRAPH_NODES_CHILD_GRAPH_NODE_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>


#include <gcxx/runtime/graph/nodes/graph_node_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

class GraphView;
class GraphExecView;

class ChildGraphNodeView : public GraphNodeView {
 public:
  GCXX_FHC ChildGraphNodeView(deviceGraphNode_t node);

  GCXX_FH auto getGraph() -> GraphView;
  GCXX_FH auto setParams(GraphExecView exec, GraphView graph) -> void;
};

GCXX_NAMESPACE_MAIN_END

// this needs to be added in gcxx/runtime/graph/graph_view.hpp to prevent the
// circular dependecy problem MAYBE modules can solve this

// #include <gcxx/runtime/details/graph/nodes/child_graph_node_view.inl>


#endif