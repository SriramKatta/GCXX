// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_NODES_MEM_FREE_NODE_VIEW_HPP_
#define GCXX_RUNTIME_GRAPH_NODES_MEM_FREE_NODE_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/graph/nodes/graph_node_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class MemFreeNodeView : public GraphNodeView {
 public:
  GCXX_FHC MemFreeNodeView(deviceGraphNode_t node);

  GCXX_FH auto getDptr() -> void*;
};

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/graph/nodes/mem_free_node_view.inl>

#endif
