#pragma once
#ifndef GCXX_RUNTIME_GRAPH_NODES_CHILD_GRAPH_NODE_VIEW_HPP_
#define GCXX_RUNTIME_GRAPH_NODES_CHILD_GRAPH_NODE_VIEW_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>


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

#endif