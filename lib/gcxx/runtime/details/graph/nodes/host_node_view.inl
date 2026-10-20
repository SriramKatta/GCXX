// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_NODES_HOST_NODE_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_NODES_HOST_NODE_VIEW_INL_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/graph/nodes/host_node_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FHC HostNodeView::HostNodeView(GraphNodeView::deviceGraphNode_t node)
    : GraphNodeView(node) {}

GCXX_FH auto HostNodeView::getParams() -> deviceHostNodeParams_t {
  deviceHostNodeParams_t params{};
  driver::graphHostNodeGetParams(m_node, &params);
  return params;
}

GCXX_FH auto HostNodeView::setParams(const deviceHostNodeParams_t& params)
  -> void {
  driver::graphHostNodeSetParams(m_node, &params);
}

GCXX_NAMESPACE_MAIN_END()

#endif
