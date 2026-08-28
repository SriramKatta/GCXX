// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_GRAPH_GRAPH_INL_
#define GCXX_RUNTIME_DETAILS_GRAPH_GRAPH_INL_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/graph/graph.hpp>
#include <gcxx/runtime/graph/graph_exec.hpp>

#include <utility>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FH auto Graph::create(const flags::graphCreate createFlag) -> Graph {
  return Graph{createFlag};
}

GCXX_FH Graph::Graph(const flags::graphCreate createFlag) GCXX_NOEXCEPT
    : GraphView(
        driver::graphCreate(static_cast<details_::flag_t>(createFlag))) {}

GCXX_FH auto Graph::destroy() -> void {
  if (m_graph != driver::INVALID_GRAPH) {
    driver::graphDestroy(m_graph);
    m_graph = driver::INVALID_GRAPH;
  }
}

GCXX_FH Graph::~Graph() GCXX_NOEXCEPT {
  destroy();
}

GCXX_FH Graph::Graph(Graph&& other) GCXX_NOEXCEPT
    : GraphView(std::exchange(other.m_graph, driver::INVALID_GRAPH)) {}

GCXX_FH auto Graph::operator=(Graph&& other) GCXX_NOEXCEPT -> Graph& {
  if (this != &other) {
    destroy();
    m_graph = std::exchange(other.m_graph, driver::INVALID_GRAPH);
  }
  return *this;
}

GCXX_FH auto Graph::release() GCXX_NOEXCEPT -> GraphView {
  auto oldGraph = m_graph;
  m_graph       = driver::INVALID_GRAPH;
  return GraphView{oldGraph};
}

GCXX_FH auto Graph::createFromRaw(deviceGraph_t graph) -> Graph {
  return Graph{graph};
}

GCXX_FH auto Graph::instantiate() const -> GraphExec {
  return GraphExec{*this};
}

GCXX_FH auto Graph::clone() const -> Graph {
  return Graph::createFromRaw(GraphView::clone().getRawHandle());
}

GCXX_NAMESPACE_MAIN_END()


#endif