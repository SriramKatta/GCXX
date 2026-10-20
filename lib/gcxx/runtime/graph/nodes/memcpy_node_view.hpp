// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_NODES_MEMCPY_NODE_VIEW_HPP_
#define GCXX_RUNTIME_GRAPH_NODES_MEMCPY_NODE_VIEW_HPP_

#include <cstddef>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/graph/nodes/graph_node_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class MemcpyNodeView : public GraphNodeView {
 public:
  using deviceMemcpy3DParams_t = driver::deviceMemcpy3DParams_t;
  using deviceMemcpyKind_t     = driver::deviceMemcpyKind_t;

  GCXX_FHC MemcpyNodeView(deviceGraphNode_t node);

  GCXX_FH auto getParams() -> deviceMemcpy3DParams_t;

  GCXX_FH auto setParams(const deviceMemcpy3DParams_t& params) -> void;

  GCXX_FH auto setParams1D(void* dst, const void* src, std::size_t count,
                           deviceMemcpyKind_t kind) -> void;
};

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/graph/nodes/memcpy_node_view.inl>

#endif
