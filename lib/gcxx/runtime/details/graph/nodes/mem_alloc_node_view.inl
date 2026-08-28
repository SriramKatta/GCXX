// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_NODES_MEM_ALLOC_NODE_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_NODES_MEM_ALLOC_NODE_VIEW_INL_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/graph/nodes/mem_alloc_node_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FHC MemAllocNodeView::MemAllocNodeView(
  GraphNodeView::deviceGraphNode_t node)
    : GraphNodeView(node) {}

GCXX_FH auto MemAllocNodeView::getParams() -> deviceMemAllocNodeParams_t {
  deviceMemAllocNodeParams_t params{};
  driver::graphMemAllocNodeGetParams(m_node, &params);
  return params;
}

GCXX_FH auto MemAllocNodeView::getDptr() -> void* {
  return getParams().dptr;
}

GCXX_NAMESPACE_MAIN_END()

#endif
