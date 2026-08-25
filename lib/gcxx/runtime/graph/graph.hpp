// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_API_RUNTIME_GRAPH_GRAPH_HPP_
#define GCXX_API_RUNTIME_GRAPH_GRAPH_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/graph/graph_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class GraphExec;

class Graph : public GraphView {
 private:
  GCXX_FH Graph(deviceGraph_t graph) GCXX_NOEXCEPT : GraphView(graph) {}


 public:
  GCXX_FH auto destroy() -> void;
  GCXX_FH Graph(const flags::graphCreate createFlag = flags::graphCreate::None)
    GCXX_NOEXCEPT;

  GCXX_FH static auto create(
    const flags::graphCreate createFlag = flags::graphCreate::None) -> Graph;


  GCXX_FH ~Graph() GCXX_NOEXCEPT;

  Graph(const Graph&)            = delete;
  Graph& operator=(const Graph&) = delete;

  GCXX_FH Graph(Graph&& other) GCXX_NOEXCEPT;
  GCXX_FH auto operator=(Graph&& other) GCXX_NOEXCEPT->Graph&;

  GCXX_FH auto release() GCXX_NOEXCEPT -> GraphView;

  GCXX_FH static auto createFromRaw(deviceGraph_t graph) -> Graph;

  GCXX_FH auto instantiate() const -> GraphExec;

  GCXX_FH auto clone() const -> Graph;
};

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/graph/graph.inl>


#endif
