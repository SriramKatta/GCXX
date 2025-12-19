#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_INL_

#include <gcxx/runtime/graph/graph.hpp>
#include <gcxx/runtime/graph/graph_exec.hpp>
#include <gcxx/runtime/runtime_error.hpp>

#include <utility>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FH auto Graph::Create(const flags::graphCreate createFlag) -> Graph {
  return Graph{createFlag};
}

GCXX_FH Graph::Graph(const flags::graphCreate createFlag) GCXX_NOEXCEPT
    : GraphView(details_::INVALID_GRAPH) {
  GCXX_SAFE_RUNTIME_CALL(GraphCreate, "Failed to create the graph", &graph_,
                         static_cast<flag_t>(createFlag));
}

GCXX_FH auto Graph::destroy() -> void {
  if (graph_ != details_::INVALID_GRAPH) {
    GCXX_SAFE_RUNTIME_CALL(GraphDestroy, "Failed to destroy the graph", graph_);
  }
}

GCXX_FH Graph::~Graph() GCXX_NOEXCEPT {
  destroy();
}

GCXX_FH Graph::Graph(Graph&& other) GCXX_NOEXCEPT
    : GraphView(std::exchange(other.graph_, details_::INVALID_GRAPH)) {}

GCXX_FH auto Graph::operator=(Graph&& other) GCXX_NOEXCEPT -> Graph& {
  if (this != &other) {
    destroy();
    graph_ = std::exchange(other.graph_, details_::INVALID_GRAPH);
  }
  return *this;
}

GCXX_FH auto Graph::Release() GCXX_NOEXCEPT -> GraphView {
  auto oldGraph = graph_;
  graph_        = details_::INVALID_GRAPH;
  return GraphView{oldGraph};
}

GCXX_FH auto Graph::CreateFromRaw(deviceGraph_t graph) -> Graph {
  return Graph{graph};
}

GCXX_FH auto Graph::Instantiate() const -> GraphExec {
  return GraphExec{*this};
}

GCXX_FH auto Graph::Clone() const -> Graph {
  return Graph::CreateFromRaw(GraphView::Clone().getRawGraph());
}

GCXX_NAMESPACE_MAIN_END


#endif