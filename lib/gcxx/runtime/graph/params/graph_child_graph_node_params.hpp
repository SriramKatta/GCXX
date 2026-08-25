// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_CHILD_GRAPH_NODE_PARAMS_HPP_
#define GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_CHILD_GRAPH_NODE_PARAMS_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime_backend/backend_graph.hpp>

// Type-state builder: setGraph() is required and may be called only once;
// build() refuses to compile otherwise.

GCXX_NAMESPACE_MAIN_BEGIN()

class GraphView;

GCXX_NAMESPACE_DETAILS_BEGIN()

// Defined in details/graph/params/graph_child_graph_node_params.inl, which is
// included at the bottom of gcxx/runtime/graph/graph_view.hpp: extracting the
// raw handle needs a complete GraphView, while GraphView's own headers need
// the params types declared here first.

GCXX_FH auto graphHandleOf(const GraphView& graph) -> driver::deviceGraph_t;

GCXX_NAMESPACE_DETAILS_END()

class ChildGraphNodeParamsView {
 public:
  using deviceChildGraphNodeParams_t = driver::deviceChildGraphNodeParams_t;
  using deviceGraph_t                = driver::deviceGraph_t;

  GCXX_FHC auto getRawParams() const -> const deviceChildGraphNodeParams_t& {
    return m_params;
  }

  GCXX_FHC auto getGraphHandle() const -> deviceGraph_t {
    return m_params.graph;
  }

  // Body lives in details/graph/params/graph_child_graph_node_params.inl.
  GCXX_FH auto getGraph() const -> GraphView;

 protected:
  GCXX_FH ChildGraphNodeParamsView() = default;

  GCXX_FHC explicit ChildGraphNodeParamsView(
    deviceChildGraphNodeParams_t params)
      : m_params{params} {}

  // Initialized once via the constructor; never mutated afterwards.
  const deviceChildGraphNodeParams_t m_params{};  // NOLINT
};

// Adds no state over the View; kept only for uniformity with the params
// kinds that do carry storage (kernel, mem-alloc, external-semaphore).
class ChildGraphNodeParams : public ChildGraphNodeParamsView {
 public:
  GCXX_FHC ChildGraphNodeParams() = default;

  GCXX_FHC explicit ChildGraphNodeParams(deviceGraph_t graph)
      : ChildGraphNodeParamsView{make_raw_params(graph)} {}

  // Body lives in details/graph/params/graph_child_graph_node_params.inl.
  GCXX_FH explicit ChildGraphNodeParams(const GraphView& graph);

  // Non-copyable/non-movable so pointers into m_params cannot dangle.
  ChildGraphNodeParams(const ChildGraphNodeParams&) = delete;
  ChildGraphNodeParams(ChildGraphNodeParams&&)      = delete;

  auto operator=(const ChildGraphNodeParams&) -> ChildGraphNodeParams& = delete;
  auto operator=(ChildGraphNodeParams&&) -> ChildGraphNodeParams&      = delete;

  ~ChildGraphNodeParams() = default;

 private:
  GCXX_FHC static auto make_raw_params(deviceGraph_t graph)
    -> deviceChildGraphNodeParams_t {
    deviceChildGraphNodeParams_t p{};
    p.graph = graph;
    return p;
  }
};

GCXX_NAMESPACE_DETAILS_BEGIN()

namespace child_graph_node_builder {

  struct graph_tag {};

}  // namespace child_graph_node_builder

namespace cgnb = child_graph_node_builder;

template <typename... Set>
class ChildGraphParamsBuilder {
 public:
  ChildGraphParamsBuilder() = default;

  GCXX_FHC auto setGraph(const GraphView& graph) const
    -> ChildGraphParamsBuilder<Set..., cgnb::graph_tag> {
    static_assert(!details_::contains_v<cgnb::graph_tag, Set...>,
                  "setGraph() may only be called once");
    ChildGraphParamsBuilder<Set..., cgnb::graph_tag> next = *this;
    next.m_graph = details_::graphHandleOf(graph);
    return next;
  }

  GCXX_FHC auto build() const -> gcxx::ChildGraphNodeParams {
    static_assert(details_::contains_v<cgnb::graph_tag, Set...>,
                  "setGraph() required before build()");
    return gcxx::ChildGraphNodeParams{m_graph};
  }

 private:
  template <typename...>
  friend class ChildGraphParamsBuilder;  // states construct each other

  // State hand-off between builder states.
  template <typename... Other>
  ChildGraphParamsBuilder(const ChildGraphParamsBuilder<Other...>& other)
      : m_graph{other.m_graph} {}

  ChildGraphNodeParamsView::deviceGraph_t m_graph{nullptr};  // NOLINT
};

GCXX_NAMESPACE_DETAILS_END()

using ChildGraphParamsBuilder_t = details_::ChildGraphParamsBuilder<>;

GCXX_FH auto ChildGraphParamsBuilder() -> ChildGraphParamsBuilder_t {
  return {};
}

GCXX_NAMESPACE_MAIN_END()


#endif
