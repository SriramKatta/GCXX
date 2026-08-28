// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_NODES_HOST_NODE_VIEW_HPP_
#define GCXX_RUNTIME_GRAPH_NODES_HOST_NODE_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/graph/nodes/graph_node_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class HostNodeView : public GraphNodeView {
 public:
  using deviceHostNodeParams_t = driver::deviceHostNodeParams_t;

  GCXX_FHC HostNodeView(deviceGraphNode_t node);

  GCXX_FH auto getParams() -> deviceHostNodeParams_t;

  GCXX_FH auto setParams(const deviceHostNodeParams_t& params) -> void;
};

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/graph/nodes/host_node_view.inl>

#endif
