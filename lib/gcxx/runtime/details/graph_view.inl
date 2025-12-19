#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_VIEW_INL_

#include <cstddef>
#include <filesystem>
#include <string_view>


#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/graph/graph_view.hpp>
#include <gcxx/runtime/runtime_error.hpp>


GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FHC GraphView::GraphView(deviceGraph_t rawgraph) : graph_(rawgraph) {}

GCXX_FHC auto GraphView::getRawGraph() const -> deviceGraph_t {
  return graph_;
}

GCXX_FH auto GraphView::SaveDotfile(std::string_view fname,
                                    flags::graphDebugDot flag) const -> void {
  // TODO : Add checks to prevent illegal file name and check folder existance
  GCXX_SAFE_RUNTIME_CALL(GraphDebugDotPrint,
                         "Failed to output the dot file of the graph", graph_,
                         fname.data(), static_cast<flag_t>(flag));
}

GCXX_FH auto GraphView::GetNumNodes() const -> size_t {
  size_t numNodes = 0;
  GCXX_SAFE_RUNTIME_CALL(GraphGetNodes, "Failed to get graph nodes", graph_,
                         nullptr, &numNodes);
  return numNodes;
}

GCXX_FH auto GraphView::GetNumEdges() const -> size_t {
  size_t numEdges = 0;
  GCXX_SAFE_RUNTIME_CALL(GraphGetEdges, "Failed to get graph edges", graph_,
                         nullptr, nullptr, &numEdges);
  return numEdges;
}

GCXX_FH auto GraphView::Clone() const -> GraphView {
  deviceGraph_t clonedGraph;
  GCXX_SAFE_RUNTIME_CALL(GraphClone, "Failed to clone graph", &clonedGraph,
                         graph_);
  return clonedGraph;
}

GCXX_NAMESPACE_MAIN_END

#include <gcxx/macros/undefine_macros.hpp>

#endif