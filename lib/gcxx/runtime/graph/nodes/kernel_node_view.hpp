// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_NODES_KERNEL_NODE_VIEW_HPP_
#define GCXX_RUNTIME_GRAPH_NODES_KERNEL_NODE_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/graph/nodes/graph_node_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class KernelNodeView : public GraphNodeView {
 public:
  using deviceKernelNodeParams_t = driver::deviceKernelNodeParams_t;

  GCXX_FHC KernelNodeView(deviceGraphNode_t node);

  GCXX_FH auto getParams() -> deviceKernelNodeParams_t;

  GCXX_FH auto setParams(const deviceKernelNodeParams_t& params) -> void;

#if GCXX_CUDA_MODE()
  using deviceKernelNodeAttrID_t    = driver::deviceKernelNodeAttrID_t;
  using deviceKernelNodeAttrValue_t = driver::deviceKernelNodeAttrValue_t;

  GCXX_FH auto getAttribute(deviceKernelNodeAttrID_t attr)
    -> deviceKernelNodeAttrValue_t;

  GCXX_FH auto setAttribute(deviceKernelNodeAttrID_t attr,
                            const deviceKernelNodeAttrValue_t& value) -> void;

  GCXX_FH auto copyAttributes(const KernelNodeView src) -> void;
#endif
};

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/graph/nodes/kernel_node_view.inl>

#endif
