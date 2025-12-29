#pragma once
#ifndef GCXX_RUNTIME_GRAPH_NODES_NODE_VIEW_HPP_
#define GCXX_RUNTIME_GRAPH_NODES_NODE_VIEW_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>


GCXX_NAMESPACE_MAIN_BEGIN


using deviceGraphNode_t = GCXX_RUNTIME_BACKEND(GraphNode_t);

class GraphNodeView {
 protected:
  deviceGraphNode_t node_;

 public:
  GCXX_FHC auto getRawNode() -> deviceGraphNode_t { return node_; }
};

GCXX_NAMESPACE_MAIN_END

#endif