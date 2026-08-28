// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_NODES_MEMCPY_NODE_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_NODES_MEMCPY_NODE_VIEW_INL_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/graph/nodes/memcpy_node_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FHC MemcpyNodeView::MemcpyNodeView(GraphNodeView::deviceGraphNode_t node)
    : GraphNodeView(node) {}

GCXX_FH auto MemcpyNodeView::getParams() -> deviceMemcpy3DParams_t {
  deviceMemcpy3DParams_t params{};
  driver::graphMemcpyNodeGetParams(m_node, &params);
  return params;
}

GCXX_FH auto MemcpyNodeView::setParams(const deviceMemcpy3DParams_t& params)
  -> void {
  driver::graphMemcpyNodeSetParams(m_node, &params);
}

GCXX_FH auto MemcpyNodeView::setParams1D(void* dst, const void* src,
                                         std::size_t count,
                                         deviceMemcpyKind_t kind) -> void {
  driver::graphMemcpyNodeSetParams1D(m_node, dst, src, count, kind);
}

GCXX_NAMESPACE_MAIN_END()

#endif
