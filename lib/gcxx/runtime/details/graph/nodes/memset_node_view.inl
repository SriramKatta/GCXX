// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_NODES_MEMSET_NODE_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_NODES_MEMSET_NODE_VIEW_INL_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/graph/nodes/memset_node_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FHC MemsetNodeView::MemsetNodeView(GraphNodeView::deviceGraphNode_t node)
    : GraphNodeView(node) {}

GCXX_FH auto MemsetNodeView::getParams() -> deviceMemsetParams_t {
  deviceMemsetParams_t params{};
  driver::graphMemsetNodeGetParams(m_node, &params);
  return params;
}

GCXX_FH auto MemsetNodeView::setParams(const deviceMemsetParams_t& params)
  -> void {
  driver::graphMemsetNodeSetParams(m_node, &params);
}

GCXX_NAMESPACE_MAIN_END()

#endif
