// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_NODES_KERNEL_NODE_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_NODES_KERNEL_NODE_VIEW_INL_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/graph/nodes/kernel_node_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FHC KernelNodeView::KernelNodeView(GraphNodeView::deviceGraphNode_t node)
    : GraphNodeView(node) {}

GCXX_FH auto KernelNodeView::getParams() -> deviceKernelNodeParams_t {
  deviceKernelNodeParams_t params{};
  driver::graphKernelNodeGetParams(m_node, &params);
  return params;
}

GCXX_FH auto KernelNodeView::setParams(const deviceKernelNodeParams_t& params)
  -> void {
  driver::graphKernelNodeSetParams(m_node, &params);
}

#if GCXX_CUDA_MODE()
GCXX_FH auto KernelNodeView::getAttribute(deviceKernelNodeAttrID_t attr)
  -> deviceKernelNodeAttrValue_t {
  deviceKernelNodeAttrValue_t value{};
  driver::graphKernelNodeGetAttribute(m_node, attr, &value);
  return value;
}

GCXX_FH auto KernelNodeView::setAttribute(
  deviceKernelNodeAttrID_t attr,
  const deviceKernelNodeAttrValue_t& value) -> void {
  driver::graphKernelNodeSetAttribute(m_node, attr, &value);
}

GCXX_FH auto KernelNodeView::copyAttributes(const KernelNodeView src) -> void {
  driver::graphKernelNodeCopyAttributes(src.m_node, m_node);
}
#endif

GCXX_NAMESPACE_MAIN_END()

#endif
