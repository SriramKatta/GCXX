// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_NODES_CHILD_GRAPH_NODE_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_NODES_CHILD_GRAPH_NODE_VIEW_INL_

#include <gcxx/internal/prologue.hpp>


#include <gcxx/runtime/graph/graph_exec_view.hpp>
#include <gcxx/runtime/graph/graph_view.hpp>


GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FHC

ChildGraphNodeView::ChildGraphNodeView(GraphNodeView::deviceGraphNode_t node)
    : GraphNodeView(node) {}

GCXX_FH auto ChildGraphNodeView::getGraph() -> GraphView {
  return {driver::graphChildGraphNodeGetGraph(node_)};
}

GCXX_FH

auto ChildGraphNodeView::setParams(GraphExecView exec,
                                   GraphView graph) -> void {
  driver::graphExecChildGraphNodeSetParams(exec.getRawExec(), node_,
                                           graph.getRawGraph());
}

GCXX_NAMESPACE_MAIN_END()

#endif