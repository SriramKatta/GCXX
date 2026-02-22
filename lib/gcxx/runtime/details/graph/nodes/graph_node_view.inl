#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_NODES_GRAPH_NODE_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_NODES_GRAPH_NODE_VIEW_INL_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/flags/graph_flags.hpp>


GCXX_NAMESPACE_MAIN_BEGIN


#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)
GCXX_FH auto GraphNodeView::geLocalId() -> unsigned int {
  unsigned int id{};
  GCXX_SAFE_RUNTIME_CALL(GraphNodeGetLocalId,
                         "Failed to query Local Id of graph node", node_, &id);
  return id;
}

GCXX_FH auto GraphNodeView::geToolsId() -> unsigned long long {
  unsigned long long id{};
  GCXX_SAFE_RUNTIME_CALL(GraphNodeGetToolsId,
                         "Failed to query Tools Id of graph node", node_, &id);
  return id;
}
#endif

GCXX_FH auto GraphNodeView::getType() -> flags::graphNodeType {
  using deviceGraphNodeType_t = GCXX_RUNTIME_BACKEND(GraphNodeType);
  deviceGraphNodeType_t enumval{};
  GCXX_SAFE_RUNTIME_CALL(
    GraphNodeGetType, "Failed to query the graph node type", node_, &enumval);
  return static_cast<flags::graphNodeType>(enumval);
}

GCXX_NAMESPACE_MAIN_END

#endif