#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_GRAPH_NODE_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_GRAPH_NODE_INL_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/graph/graph_node.hpp>


GCXX_NAMESPACE_MAIN_BEGIN

// ════════════════════════════════════════════════════════════════════════════
// KernelNodeView Implementation
// ════════════════════════════════════════════════════════════════════════════

GCXX_FH auto KernelNodeView::GetParams(
  details_::deviceKernelNodeParams_t* params) const -> const KernelNodeView& {
  GCXX_SAFE_RUNTIME_CALL(GraphKernelNodeGetParams,
                         "Failed to get kernel node parameters", node_, params);
  return *this;
}

GCXX_FH auto KernelNodeView::GetParams() const -> KernelNodeParams {
  KernelNodeParams result;
  GetParams(result.getRawParams());
  return result;
}

GCXX_FH auto KernelNodeView::SetParams(
  const details_::deviceKernelNodeParams_t* params) -> KernelNodeView& {
  GCXX_SAFE_RUNTIME_CALL(GraphKernelNodeSetParams,
                         "Failed to set kernel node parameters", node_, params);
  return *this;
}

GCXX_FH auto KernelNodeView::SetParams(const KernelNodeParams& params)
  -> KernelNodeView& {
  return SetParams(params.getRawParams());
}

GCXX_FH auto KernelNodeView::GetAttribute(
  details_::deviceKernelNodeAttrID attr,
  details_::deviceKernelNodeAttrValue* value) const -> const KernelNodeView& {
  GCXX_SAFE_RUNTIME_CALL(GraphKernelNodeGetAttribute,
                         "Failed to get kernel node attribute", node_, attr,
                         value);
  return *this;
}

GCXX_FH auto KernelNodeView::SetAttribute(
  details_::deviceKernelNodeAttrID attr,
  const details_::deviceKernelNodeAttrValue* value) -> KernelNodeView& {
  GCXX_SAFE_RUNTIME_CALL(GraphKernelNodeSetAttribute,
                         "Failed to set kernel node attribute", node_, attr,
                         value);
  return *this;
}

GCXX_FH auto KernelNodeView::CopyAttributesFrom(const KernelNodeView& src)
  -> KernelNodeView& {
  GCXX_SAFE_RUNTIME_CALL(GraphKernelNodeCopyAttributes,
                         "Failed to copy kernel node attributes", node_,
                         src.node_);
  return *this;
}

GCXX_FH auto KernelNodeView::SetParamsInExec(
  details_::deviceGraphExec_t exec,
  const details_::deviceKernelNodeParams_t* params) -> KernelNodeView& {
  GCXX_SAFE_RUNTIME_CALL(GraphExecKernelNodeSetParams,
                         "Failed to set kernel node parameters in graph exec",
                         exec, node_, params);
  return *this;
}

GCXX_FH auto KernelNodeView::SetParamsInExec(details_::deviceGraphExec_t exec,
                                             const KernelNodeParams& params)
  -> KernelNodeView& {
  return SetParamsInExec(exec, params.getRawParams());
}

#if GCXX_CUDA_MODE
// ════════════════════════════════════════════════════════════════════════════
// Device API Implementation (CUDA only)
// ════════════════════════════════════════════════════════════════════════════

GCXX_FD auto KernelNodeView::SetEnabled(
  details_::deviceGraphDeviceNode_t deviceNode, bool enable) -> void {
  GCXX_RUNTIME_BACKEND(GraphKernelNodeSetEnabled)(deviceNode, enable);
}

GCXX_FD auto KernelNodeView::SetGridDim(
  details_::deviceGraphDeviceNode_t deviceNode, dim3 gridDim) -> void {
  GCXX_RUNTIME_BACKEND(GraphKernelNodeSetGridDim)(deviceNode, gridDim);
}

GCXX_FD auto KernelNodeView::SetParam(
  details_::deviceGraphDeviceNode_t deviceNode, size_t offset,
  const void* value, size_t size) -> void {
  GCXX_RUNTIME_BACKEND(GraphKernelNodeSetParam)
  (deviceNode, offset, value, size);
}

GCXX_FD auto KernelNodeView::ApplyUpdates(
  const details_::deviceGraphKernelNodeUpdate* updates, size_t updateCount)
  -> void {
  GCXX_RUNTIME_BACKEND(GraphKernelNodeUpdatesApply)(updates, updateCount);
}
#endif

GCXX_NAMESPACE_MAIN_END

#endif
