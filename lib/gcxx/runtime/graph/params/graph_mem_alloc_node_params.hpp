// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_MEM_ALLOC_NODE_PARAMS_HPP_
#define GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_MEM_ALLOC_NODE_PARAMS_HPP_

#include <cstddef>
#include <utility>
#include <vector>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime/memory/spans/spans.hpp>
#include <gcxx/runtime_backend/backend_graph.hpp>

// Type-state builder: setPoolProps() and setBytesize() are required and may
// be called only once; build() refuses to compile otherwise. setAccessDescs()
// is optional (empty = no peer access). The output dptr is filled by the
// driver when the node is created, not by the caller.
//
// Members touching the owning storage (std::vector) are host-only GCXX_FH:
// the owning params are not literal types, so they cannot be constexpr.

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_DETAILS_BEGIN()

// Access-descriptor storage, inherited BEFORE MemAllocNodeParamsView so the
// raw params pointer can point into it safely.
struct MemAccessDescStore {
  std::vector<driver::deviceMemAccessDesc_t> descs{};  // NOLINT

  MemAccessDescStore() = default;

  MemAccessDescStore(const driver::deviceMemAccessDesc_t* first,
                     const driver::deviceMemAccessDesc_t* last)
      : descs(first, last) {}
};

GCXX_NAMESPACE_DETAILS_END()

class MemAllocNodeParamsView {
 public:
  using deviceMemAllocNodeParams_t = driver::deviceMemAllocNodeParams_t;
  using deviceMemPoolProps_t       = driver::deviceMemPoolProps_t;
  using deviceMemAccessDesc_t      = driver::deviceMemAccessDesc_t;

  GCXX_FHC auto getRawParams() const -> const deviceMemAllocNodeParams_t& {
    return m_params;
  }

  GCXX_FHC auto getPoolProps() const -> const deviceMemPoolProps_t& {
    return m_params.poolProps;
  }

  GCXX_FHC auto getAccessDescs() const -> const deviceMemAccessDesc_t* {
    return m_params.accessDescs;
  }

  GCXX_FHC auto getAccessDescCount() const -> std::size_t {
    return m_params.accessDescCount;
  }

  GCXX_FHC auto getBytesize() const -> std::size_t { return m_params.bytesize; }

  // Output-only: address of the allocation, filled in by the driver when the
  // alloc node is created.
  GCXX_FHC auto getDptr() const -> void* { return m_params.dptr; }

 protected:
  GCXX_FH MemAllocNodeParamsView() = default;

  GCXX_FHC explicit MemAllocNodeParamsView(deviceMemAllocNodeParams_t params)
      : m_params{params} {}

  // Initialized once via the constructor; never mutated afterwards.
  const deviceMemAllocNodeParams_t m_params{};  // NOLINT
};

class MemAllocNodeParams : private details_::MemAccessDescStore,
                           public MemAllocNodeParamsView {
 public:
  MemAllocNodeParams() = default;

  MemAllocNodeParams(const deviceMemPoolProps_t& poolProps,
                     std::size_t bytesize)
      : MemAllocNodeParams{poolProps, gcxx::span<const deviceMemAccessDesc_t>{},
                           bytesize} {}

  MemAllocNodeParams(const deviceMemPoolProps_t& poolProps,
                     gcxx::span<const deviceMemAccessDesc_t> accessDescs,
                     std::size_t bytesize)
      : details_::MemAccessDescStore{accessDescs.data(),
                                     accessDescs.data() + accessDescs.size()},
        MemAllocNodeParamsView{make_raw_params(
          poolProps, details_::MemAccessDescStore::descs.data(),
          details_::MemAccessDescStore::descs.size(), bytesize)} {}

  // Non-copyable/non-movable so accessDescs cannot dangle to an old object.
  MemAllocNodeParams(const MemAllocNodeParams&) = delete;
  MemAllocNodeParams(MemAllocNodeParams&&)      = delete;

  auto operator=(const MemAllocNodeParams&) -> MemAllocNodeParams& = delete;
  auto operator=(MemAllocNodeParams&&) -> MemAllocNodeParams&      = delete;

  ~MemAllocNodeParams() = default;

 private:
  GCXX_FHC static auto make_raw_params(
    const deviceMemPoolProps_t& poolProps,
    const deviceMemAccessDesc_t* accessDescs, std::size_t accessDescCount,
    std::size_t bytesize) -> deviceMemAllocNodeParams_t {
    deviceMemAllocNodeParams_t p{};
    p.poolProps       = poolProps;
    p.accessDescs     = accessDescs;
    p.accessDescCount = accessDescCount;
    p.bytesize        = bytesize;
    p.dptr            = nullptr;
    return p;
  }
};

GCXX_NAMESPACE_DETAILS_BEGIN()

namespace mem_alloc_node_builder {

  struct pool_tag {};
  struct size_tag {};

}  // namespace mem_alloc_node_builder

namespace manb = mem_alloc_node_builder;

template <typename... Set>
class MemAllocParamsBuilder {
 public:
  MemAllocParamsBuilder() = default;

  GCXX_FH auto setPoolProps(
    const MemAllocNodeParamsView::deviceMemPoolProps_t& poolProps) const
    -> MemAllocParamsBuilder<Set..., manb::pool_tag> {
    static_assert(!details_::contains_v<manb::pool_tag, Set...>,
                  "setPoolProps() may only be called once");
    MemAllocParamsBuilder<Set..., manb::pool_tag> next = *this;
    next.m_poolProps                                   = poolProps;
    return next;
  }

  GCXX_FH auto setBytesize(std::size_t bytesize) const
    -> MemAllocParamsBuilder<Set..., manb::size_tag> {
    static_assert(!details_::contains_v<manb::size_tag, Set...>,
                  "setBytesize() may only be called once");
    MemAllocParamsBuilder<Set..., manb::size_tag> next = *this;
    next.m_bytesize                                    = bytesize;
    return next;
  }

  // Optional; empty by default (no peer access).
  GCXX_FH auto setAccessDescs(
    gcxx::span<const MemAllocNodeParamsView::deviceMemAccessDesc_t> accessDescs)
    const -> MemAllocParamsBuilder {
    MemAllocParamsBuilder next = *this;
    next.m_accessDescs.assign(accessDescs.data(),
                              accessDescs.data() + accessDescs.size());
    return next;
  }

  GCXX_FH auto build() const -> gcxx::MemAllocNodeParams {
    static_assert(details_::contains_v<manb::pool_tag, Set...>,
                  "setPoolProps() required before build()");
    static_assert(details_::contains_v<manb::size_tag, Set...>,
                  "setBytesize() required before build()");
    return gcxx::MemAllocNodeParams{
      m_poolProps,
      gcxx::span<const MemAllocNodeParamsView::deviceMemAccessDesc_t>{
        m_accessDescs.data(), m_accessDescs.size()},
      m_bytesize};
  }

 private:
  template <typename...>
  friend class MemAllocParamsBuilder;  // states construct each other

  // State hand-off between builder states.
  template <typename... Other>
  MemAllocParamsBuilder(const MemAllocParamsBuilder<Other...>& other)
      : m_poolProps{other.m_poolProps},
        m_accessDescs{other.m_accessDescs},
        m_bytesize{other.m_bytesize} {}

  MemAllocNodeParamsView::deviceMemPoolProps_t m_poolProps{};
  std::vector<MemAllocNodeParamsView::deviceMemAccessDesc_t> m_accessDescs{};
  std::size_t m_bytesize{0};
};

GCXX_NAMESPACE_DETAILS_END()

using MemAllocParamsBuilder_t = details_::MemAllocParamsBuilder<>;

GCXX_FH auto MemAllocParamsBuilder() -> MemAllocParamsBuilder_t {
  return {};
}

GCXX_NAMESPACE_MAIN_END()


#endif
