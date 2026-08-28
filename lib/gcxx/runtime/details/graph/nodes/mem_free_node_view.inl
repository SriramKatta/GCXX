// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_NODES_MEM_FREE_NODE_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_NODES_MEM_FREE_NODE_VIEW_INL_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/graph/nodes/mem_free_node_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FHC MemFreeNodeView::MemFreeNodeView(GraphNodeView::deviceGraphNode_t node)
    : GraphNodeView(node) {}

GCXX_FH auto MemFreeNodeView::getDptr() -> void* {
  void* dptr{nullptr};
  driver::graphMemFreeNodeGetParams(m_node, &dptr);
  return dptr;
}

GCXX_NAMESPACE_MAIN_END()

#endif
