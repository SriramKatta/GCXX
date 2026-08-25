// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_NODES_EVENT_RECORD_NODE_VIEW_HPP_
#define GCXX_RUNTIME_GRAPH_NODES_EVENT_RECORD_NODE_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/event/event_view.hpp>
#include <gcxx/runtime/graph/nodes/graph_node_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class EventRecordNodeView : public GraphNodeView {
 public:
  GCXX_FHC EventRecordNodeView(deviceGraphNode_t node);

  GCXX_FH auto getEvent() -> EventView;

  GCXX_FH auto setEvent(const EventView event) -> void;
};

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/graph/nodes/event_record_node_view.inl>

#endif
