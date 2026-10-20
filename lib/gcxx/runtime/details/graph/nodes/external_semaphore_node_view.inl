// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_NODES_EXTERNAL_SEMAPHORE_NODE_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_NODES_EXTERNAL_SEMAPHORE_NODE_VIEW_INL_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/graph/nodes/external_semaphore_node_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FHC ExternalSemaphoreSignalNodeView::ExternalSemaphoreSignalNodeView(
  GraphNodeView::deviceGraphNode_t node)
    : GraphNodeView(node) {}

GCXX_FH auto ExternalSemaphoreSignalNodeView::getParams()
  -> deviceExternalSemaphoreSignalNodeParams_t {
  deviceExternalSemaphoreSignalNodeParams_t params{};
  driver::graphExternalSemaphoresSignalNodeGetParams(m_node, &params);
  return params;
}

GCXX_FH auto ExternalSemaphoreSignalNodeView::setParams(
  const deviceExternalSemaphoreSignalNodeParams_t& params) -> void {
  driver::graphExternalSemaphoresSignalNodeSetParams(m_node, &params);
}

GCXX_FHC ExternalSemaphoreWaitNodeView::ExternalSemaphoreWaitNodeView(
  GraphNodeView::deviceGraphNode_t node)
    : GraphNodeView(node) {}

GCXX_FH auto ExternalSemaphoreWaitNodeView::getParams()
  -> deviceExternalSemaphoreWaitNodeParams_t {
  deviceExternalSemaphoreWaitNodeParams_t params{};
  driver::graphExternalSemaphoresWaitNodeGetParams(m_node, &params);
  return params;
}

GCXX_FH auto ExternalSemaphoreWaitNodeView::setParams(
  const deviceExternalSemaphoreWaitNodeParams_t& params) -> void {
  driver::graphExternalSemaphoresWaitNodeSetParams(m_node, &params);
}

GCXX_NAMESPACE_MAIN_END()

#endif
