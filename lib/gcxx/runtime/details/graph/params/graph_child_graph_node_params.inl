// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_PARAMS_GRAPH_CHILD_GRAPH_NODE_PARAMS_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_PARAMS_GRAPH_CHILD_GRAPH_NODE_PARAMS_INL_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/graph/graph_view.hpp>
#include <gcxx/runtime/graph/params/graph_child_graph_node_params.hpp>

// Method definitions live here, included at the bottom of
// gcxx/runtime/graph/graph_view.hpp (not from the params header).
// ChildGraphNodeParams{GraphView} and getGraph() need a complete GraphView,
// while GraphView's own headers need the params types declared first.

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_DETAILS_BEGIN()

GCXX_FH auto graphHandleOf(const GraphView& graph) -> driver::deviceGraph_t {
  return graph.getRawHandle();
}

GCXX_NAMESPACE_DETAILS_END()

GCXX_FH auto ChildGraphNodeParamsView::getGraph() const -> GraphView {
  return GraphView{m_params.graph};
}

GCXX_FH ChildGraphNodeParams::ChildGraphNodeParams(const GraphView& graph)
    : ChildGraphNodeParamsView{
        make_raw_params(details_::graphHandleOf(graph))} {}

GCXX_NAMESPACE_MAIN_END()

#endif
