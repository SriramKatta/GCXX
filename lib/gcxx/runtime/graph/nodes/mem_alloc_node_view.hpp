// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_NODES_MEM_ALLOC_NODE_VIEW_HPP_
#define GCXX_RUNTIME_GRAPH_NODES_MEM_ALLOC_NODE_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/graph/nodes/graph_node_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class MemAllocNodeView : public GraphNodeView {
 public:
  using deviceMemAllocNodeParams_t = driver::deviceMemAllocNodeParams_t;

  GCXX_FHC MemAllocNodeView(deviceGraphNode_t node);

  GCXX_FH auto getParams() -> deviceMemAllocNodeParams_t;

  // Address of the allocation owned by the node; valid once the node has
  // been created (filled in by the driver).
  GCXX_FH auto getDptr() -> void*;
};

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/graph/nodes/mem_alloc_node_view.inl>

#endif
