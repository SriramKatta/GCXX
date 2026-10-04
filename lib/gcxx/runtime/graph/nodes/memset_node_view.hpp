// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_NODES_MEMSET_NODE_VIEW_HPP_
#define GCXX_RUNTIME_GRAPH_NODES_MEMSET_NODE_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/graph/nodes/graph_node_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class MemsetNodeView : public GraphNodeView {
 public:
  using deviceMemsetParams_t = driver::deviceMemsetParams_t;

  GCXX_FHC MemsetNodeView(deviceGraphNode_t node);

  GCXX_FH auto getParams() -> deviceMemsetParams_t;

  GCXX_FH auto setParams(const deviceMemsetParams_t& params) -> void;
};

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/graph/nodes/memset_node_view.inl>

#endif
