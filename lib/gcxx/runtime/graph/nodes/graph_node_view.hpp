#pragma once
#ifndef GCXX_RUNTIME_GRAPH_NODES_NODE_VIEW_HPP_
#define GCXX_RUNTIME_GRAPH_NODES_NODE_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/flags/graph_flags.hpp>


GCXX_NAMESPACE_MAIN_BEGIN

class GraphView;
class GraphExecView;


using deviceGraphNode_t = GCXX_RUNTIME_BACKEND(GraphNode_t);

// TODO : needs diffrent specilaization for memallocnode, memfreenode ..... and
// have the set* and get* menber funtions that can be speciliazed to set the
// appropriate params like

class GraphNodeView {
 protected:
  deviceGraphNode_t node_;  // NOLINT

 public:
  GCXX_FHC GraphNodeView(deviceGraphNode_t node) : node_(node) {}

  GCXX_FHC auto getRawNode() -> deviceGraphNode_t { return node_; }

  GCXX_FH auto getContainingGraph() -> GraphView;

  GCXX_FH auto getType() -> flags::graphNodeType;

#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)

  GCXX_FH auto geLocalId() -> unsigned int;

  GCXX_FH auto geToolsId() -> unsigned long long;

#endif
};

GCXX_NAMESPACE_MAIN_END

#include <gcxx/runtime/details/graph/nodes/graph_node_view.inl>

#endif