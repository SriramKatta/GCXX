// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_GRAPH_NODE_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_GRAPH_NODE_INL_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/graph/graph_node.hpp>
#include <gcxx/runtime_backend/backend_graph.hpp>


GCXX_NAMESPACE_MAIN_BEGIN()

// ════════════════════════════════════════════════════════════════════════════
// KernelNodeView Implementation
// ════════════════════════════════════════════════════════════════════════════

GCXX_FH auto KernelNodeView::GetParams(
  details_::deviceKernelNodeParams_t* params) const -> const KernelNodeView& {
  driver::graphKernelNodeGetParams(node_, params);
  return *this;
}

GCXX_FH auto KernelNodeView::GetParams() const -> KernelNodeParams {
  KernelNodeParams result;
  GetParams(result.getRawParams());
  return result;
}

GCXX_FH auto KernelNodeView::SetParams(
  const details_::deviceKernelNodeParams_t* params) -> KernelNodeView& {
  driver::graphKernelNodeSetParams(node_, params);
  return *this;
}

GCXX_FH auto KernelNodeView::SetParams(const KernelNodeParams& params)
  -> KernelNodeView& {
  return SetParams(params.getRawParams());
}

GCXX_FH auto KernelNodeView::GetAttribute(
  details_::deviceKernelNodeAttrID attr,
  details_::deviceKernelNodeAttrValue* value) const -> const KernelNodeView& {
  driver::graphKernelNodeGetAttribute(node_, attr, value);
  return *this;
}

GCXX_FH auto KernelNodeView::SetAttribute(
  details_::deviceKernelNodeAttrID attr,
  const details_::deviceKernelNodeAttrValue* value) -> KernelNodeView& {
  driver::graphKernelNodeSetAttribute(node_, attr, value);
  return *this;
}

GCXX_FH auto KernelNodeView::CopyAttributesFrom(const KernelNodeView& src)
  -> KernelNodeView& {
  driver::graphKernelNodeCopyAttributes(src.node_, node_);
  return *this;
}

GCXX_FH auto KernelNodeView::SetParamsInExec(
  details_::deviceGraphExec_t exec,
  const details_::deviceKernelNodeParams_t* params) -> KernelNodeView& {
  driver::graphExecKernelNodeSetParams(exec, node_, params);
  return *this;
}

GCXX_FH auto KernelNodeView::SetParamsInExec(details_::deviceGraphExec_t exec,
                                             const KernelNodeParams& params)
  -> KernelNodeView& {
  return SetParamsInExec(exec, params.getRawParams());
}
#if GCXX_CUDA_MODE()
// ════════════════════════════════════════════════════════════════════════════
// Device API Implementation (CUDA only)
// ════════════════════════════════════════════════════════════════════════════

GCXX_FD
auto KernelNodeView::SetEnabled(details_::deviceGraphDeviceNode_t deviceNode,
                                bool enable) -> void {
  driver::graphKernelNodeSetEnabled(deviceNode, enable);
}

GCXX_FD
auto KernelNodeView::SetGridDim(details_::deviceGraphDeviceNode_t deviceNode,
                                dim3 gridDim) -> void {
  driver::graphKernelNodeSetGridDim(deviceNode, gridDim);
}

GCXX_FD
auto KernelNodeView::SetParam(details_::deviceGraphDeviceNode_t deviceNode,
                              size_t offset, const void* value, size_t size)
  -> void {
  driver::graphKernelNodeSetParam(deviceNode, offset, value, size);
}

GCXX_FD
auto KernelNodeView::ApplyUpdates(
  const details_::deviceGraphKernelNodeUpdate* updates, size_t updateCount)
  -> void {
  driver::graphKernelNodeUpdatesApply(updates, updateCount);
}
#endif

GCXX_NAMESPACE_MAIN_END()

#endif
