// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_NODES_NODE_VIEW_HPP_
#define GCXX_RUNTIME_GRAPH_NODES_NODE_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/flags/graph_flags.hpp>
#include <gcxx/runtime_backend/backend_graph.hpp>


GCXX_NAMESPACE_MAIN_BEGIN()

class GraphView;
class GraphExecView;

class GraphNodeView {
 public:
  using deviceGraphNode_t = driver::deviceGraphNode_t;
  using raw_handle_type   = driver::deviceGraphNode_t;

  GCXX_FHC GraphNodeView(deviceGraphNode_t node) : m_node(node) {}

  GCXX_FHC auto getRawHandle() const -> deviceGraphNode_t { return m_node; }

  GCXX_FH auto getContainingGraph() -> GraphView;

  GCXX_FH auto getType() -> flags::graphNodeType;

#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)

  GCXX_FH auto getLocalId() -> unsigned int;

  GCXX_FH auto getToolsId() -> unsigned long long;

#endif

 protected:
  deviceGraphNode_t m_node;  // NOLINT
};

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/graph/nodes/graph_node_view.inl>

#endif