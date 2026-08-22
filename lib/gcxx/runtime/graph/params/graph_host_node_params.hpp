// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_HOST_NODE_PARAMS_HPP_
#define GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_HOST_NODE_PARAMS_HPP_

#include <cstddef>
#include <cstring>
#include <utility>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime_backend/backend_graph.hpp>

#include <gcxx/runtime/memory/memory_helpers.hpp>

// Type-state builder: setHostCallbackFn() is required and may be called only
// once; build() refuses to compile otherwise.

GCXX_NAMESPACE_MAIN_BEGIN()

class HostNodeParamsView {
 public:
  using deviceHostCallBackFn_t = driver::deviceHostCallBackFn_t;
  using deviceHostNodeParams_t = driver::deviceHostNodeParams_t;

  GCXX_FHC auto getRawParams() const -> const deviceHostNodeParams_t& {
    return m_params;
  }

  GCXX_FHC auto getHostFunc() const -> const deviceHostCallBackFn_t {
    return m_params.fn;
  }

  GCXX_FHC auto getUserData() const -> const void* { return m_params.userData; }

 protected:
  GCXX_FH HostNodeParamsView() = default;

  GCXX_FHC explicit HostNodeParamsView(deviceHostNodeParams_t params)
      : m_params{params} {}

  // Initialized once via the constructor; never mutated afterwards.
  const deviceHostNodeParams_t m_params{};  // NOLINT
};

class HostNodeParams : public HostNodeParamsView {
 public:
  GCXX_FHC HostNodeParams() = default;

  GCXX_FHC HostNodeParams(deviceHostCallBackFn_t fn, void* udata)
      : HostNodeParamsView{make_raw_params(fn, udata)} {}

  // Non-copyable/non-movable so pointers into m_params cannot dangle.
  HostNodeParams(const HostNodeParams&) = delete;
  HostNodeParams(HostNodeParams&&)      = delete;

  auto operator=(const HostNodeParams&) -> HostNodeParams& = delete;
  auto operator=(HostNodeParams&&) -> HostNodeParams&      = delete;

  ~HostNodeParams() = default;

 private:
  GCXX_FHC static auto make_raw_params(deviceHostCallBackFn_t fn,
                                       void* udata) -> deviceHostNodeParams_t {
    deviceHostNodeParams_t p{};
    p.fn       = fn;
    p.userData = udata;
    return p;
  }
};

GCXX_NAMESPACE_DETAILS_BEGIN()

namespace host_node_builder {

  struct callback_tag {};

}  // namespace host_node_builder

namespace hnb = host_node_builder;

template <typename... Set>
class HostNodeParamsBuilder {
 public:
  HostNodeParamsBuilder() = default;

  GCXX_FHC auto setHostCallbackFn(
    HostNodeParamsView::deviceHostCallBackFn_t func) const
    -> HostNodeParamsBuilder<Set..., hnb::callback_tag> {
    static_assert(!details_::contains_v<hnb::callback_tag, Set...>,
                  "setHostCallbackFn() may only be called once");
    HostNodeParamsBuilder<Set..., hnb::callback_tag> next = *this;
    next.m_fn                                             = func;
    return next;
  }

  GCXX_FHC auto setUserData(void* udata) const -> HostNodeParamsBuilder {
    HostNodeParamsBuilder next = *this;
    next.m_udata               = udata;
    return next;
  }

  GCXX_FHC auto build() const -> gcxx::HostNodeParams {
    static_assert(details_::contains_v<hnb::callback_tag, Set...>,
                  "setHostCallbackFn() required before build()");
    return gcxx::HostNodeParams{m_fn, m_udata};
  }

 private:
  template <typename...>
  friend class HostNodeParamsBuilder;  // states construct each other

  // State hand-off between builder states.
  template <typename... Other>
  HostNodeParamsBuilder(const HostNodeParamsBuilder<Other...>& other)
      : m_fn{other.m_fn}, m_udata{other.m_udata} {}

  HostNodeParamsView::deviceHostCallBackFn_t m_fn{nullptr};
  void* m_udata{nullptr};
};

GCXX_NAMESPACE_DETAILS_END()

GCXX_FH auto HostNodeParamsBuilder() -> details_::HostNodeParamsBuilder<> {
  return {};
}

GCXX_NAMESPACE_MAIN_END()


#endif
