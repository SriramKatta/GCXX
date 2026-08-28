// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_NODES_EXTERNAL_SEMAPHORE_NODE_VIEW_HPP_
#define GCXX_RUNTIME_GRAPH_NODES_EXTERNAL_SEMAPHORE_NODE_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/graph/nodes/graph_node_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class ExternalSemaphoreSignalNodeView : public GraphNodeView {
 public:
  using deviceExternalSemaphoreSignalNodeParams_t =
    driver::deviceExternalSemaphoreSignalNodeParams_t;

  GCXX_FHC ExternalSemaphoreSignalNodeView(deviceGraphNode_t node);

  GCXX_FH auto getParams() -> deviceExternalSemaphoreSignalNodeParams_t;

  GCXX_FH auto setParams(
    const deviceExternalSemaphoreSignalNodeParams_t& params) -> void;
};

class ExternalSemaphoreWaitNodeView : public GraphNodeView {
 public:
  using deviceExternalSemaphoreWaitNodeParams_t =
    driver::deviceExternalSemaphoreWaitNodeParams_t;

  GCXX_FHC ExternalSemaphoreWaitNodeView(deviceGraphNode_t node);

  GCXX_FH auto getParams() -> deviceExternalSemaphoreWaitNodeParams_t;

  GCXX_FH auto setParams(const deviceExternalSemaphoreWaitNodeParams_t& params)
    -> void;
};

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/graph/nodes/external_semaphore_node_view.inl>

#endif
