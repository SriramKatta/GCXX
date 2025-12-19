#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_INL_

#include <gcxx/runtime/graph/graph.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FH Graph::Graph(const flags::graphCreate createFlag) GCXX_NOEXCEPT {
  GCXX_SAFE_RUNTIME_CALL(GraphCreate, "Failed to create the graph", &graph_, 0);
}

GCXX_FH Graph::~Graph() GCXX_NOEXCEPT {
  GCXX_SAFE_RUNTIME_CALL(GraphDestroy, "Failed to destroy the graph", graph_);
}


GCXX_FH  auto Graph::CreateFromRaw(deviceGraph_t graph)->Graph{
    return {graph};
  }

GCXX_NAMESPACE_MAIN_END


#endif