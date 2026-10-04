// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_MEM_FREE_NODE_PARAMS_HPP_
#define GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_MEM_FREE_NODE_PARAMS_HPP_

#include <cstddef>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime_backend/backend_graph.hpp>

// Type-state builder: setDptr() is required and may be called only once;
// build() refuses to compile otherwise.

GCXX_NAMESPACE_MAIN_BEGIN()

class MemFreeNodeParamsView {
 public:
  using deviceMemFreeNodeParams_t = driver::deviceMemFreeNodeParams_t;

  GCXX_FHC auto getRawParams() const -> const deviceMemFreeNodeParams_t& {
    return m_params;
  }

  GCXX_FHC auto getDptr() const -> void* { return m_params.dptr; }

 protected:
  GCXX_FH MemFreeNodeParamsView() = default;

  GCXX_FHC explicit MemFreeNodeParamsView(deviceMemFreeNodeParams_t params)
      : m_params{params} {}

  // Initialized once via the constructor; never mutated afterwards.
  const deviceMemFreeNodeParams_t m_params{};  // NOLINT
};

// Adds no state over the View; kept only for uniformity with the params
// kinds that do carry storage (kernel, mem-alloc, external-semaphore).
class MemFreeNodeParams : public MemFreeNodeParamsView {
 public:
  GCXX_FHC MemFreeNodeParams() = default;

  GCXX_FHC explicit MemFreeNodeParams(void* dptr)
      : MemFreeNodeParamsView{make_raw_params(dptr)} {}

  // Non-copyable/non-movable so pointers into m_params cannot dangle.
  MemFreeNodeParams(const MemFreeNodeParams&) = delete;
  MemFreeNodeParams(MemFreeNodeParams&&)      = delete;

  auto operator=(const MemFreeNodeParams&) -> MemFreeNodeParams& = delete;
  auto operator=(MemFreeNodeParams&&) -> MemFreeNodeParams&      = delete;

  ~MemFreeNodeParams() = default;

 private:
  GCXX_FHC static auto make_raw_params(void* dptr)
    -> deviceMemFreeNodeParams_t {
    deviceMemFreeNodeParams_t p{};
    p.dptr = dptr;
    return p;
  }
};

GCXX_NAMESPACE_DETAILS_BEGIN()

namespace mem_free_node_builder {

  struct dptr_tag {};

}  // namespace mem_free_node_builder

namespace mfnb = mem_free_node_builder;

template <typename... Set>
class MemFreeParamsBuilder {
 public:
  MemFreeParamsBuilder() = default;

  GCXX_FHC auto setDptr(void* dptr) const
    -> MemFreeParamsBuilder<Set..., mfnb::dptr_tag> {
    static_assert(!details_::contains_v<mfnb::dptr_tag, Set...>,
                  "setDptr() may only be called once");
    MemFreeParamsBuilder<Set..., mfnb::dptr_tag> next = *this;
    next.m_dptr                                       = dptr;
    return next;
  }

  GCXX_FHC auto build() const -> gcxx::MemFreeNodeParams {
    static_assert(details_::contains_v<mfnb::dptr_tag, Set...>,
                  "setDptr() required before build()");
    return gcxx::MemFreeNodeParams{m_dptr};
  }

 private:
  template <typename...>
  friend class MemFreeParamsBuilder;  // states construct each other

  // State hand-off between builder states.
  template <typename... Other>
  MemFreeParamsBuilder(const MemFreeParamsBuilder<Other...>& other)
      : m_dptr{other.m_dptr} {}

  void* m_dptr{nullptr};
};

GCXX_NAMESPACE_DETAILS_END()

using MemFreeParamsBuilder_t = details_::MemFreeParamsBuilder<>;

GCXX_FH auto MemFreeParamsBuilder() -> MemFreeParamsBuilder_t {
  return {};
}

GCXX_NAMESPACE_MAIN_END()


#endif
