// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_NODES_EVENT_WAIT_NODE_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_NODES_EVENT_WAIT_NODE_VIEW_INL_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/graph/nodes/event_wait_node_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FHC
EventWaitNodeView::EventWaitNodeView(GraphNodeView::deviceGraphNode_t node)
    : GraphNodeView(node) {}

GCXX_FH auto EventWaitNodeView::getEvent() -> EventView {
  return {driver::graphEventWaitNodeGetEvent(m_node)};
}

GCXX_FH auto EventWaitNodeView::setEvent(const EventView& event) -> void {
  driver::graphEventWaitNodeSetEvent(m_node, event.getRawHandle());
}

GCXX_NAMESPACE_MAIN_END()

#endif
