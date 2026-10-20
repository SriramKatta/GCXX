// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_EXTERNAL_SEMAPHORE_NODE_PARAMS_HPP_
#define GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_EXTERNAL_SEMAPHORE_NODE_PARAMS_HPP_

#include <cstddef>
#include <utility>
#include <vector>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime/memory/spans/spans.hpp>
#include <gcxx/runtime_backend/backend_graph.hpp>

// Type-state builders: setSemaphores() and setSignalParams()/setWaitParams()
// are required and may be called only once; build() refuses to compile
// otherwise. Both arrays must have the same length (checked at construction).
//
// Members touching the owning storage (std::vector) are host-only GCXX_FH:
// the owning params are not literal types, so they cannot be constexpr.

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_DETAILS_BEGIN()

// Semaphore/parameter storage, inherited BEFORE the params views so the raw
// params pointers can point into it safely.
template <typename SemaphoreParamsT>
struct ExtSemStore {
  std::vector<driver::deviceExternalSemaphore_t> sems{};  // NOLINT
  std::vector<SemaphoreParamsT> params{};                 // NOLINT

  ExtSemStore() = default;

  ExtSemStore(const driver::deviceExternalSemaphore_t* semsFirst,
              const driver::deviceExternalSemaphore_t* semsLast,
              const SemaphoreParamsT* paramsFirst,
              const SemaphoreParamsT* paramsLast)
      : sems(semsFirst, semsLast), params(paramsFirst, paramsLast) {}
};

GCXX_NAMESPACE_DETAILS_END()

class ExternalSemaphoreSignalNodeParamsView {
 public:
  using deviceExternalSemaphoreSignalNodeParams_t =
    driver::deviceExternalSemaphoreSignalNodeParams_t;
  using deviceExternalSemaphore_t = driver::deviceExternalSemaphore_t;
  using deviceExternalSemaphoreSignalParams_t =
    driver::deviceExternalSemaphoreSignalParams_t;

  GCXX_FHC auto getRawParams() const
    -> const deviceExternalSemaphoreSignalNodeParams_t& {
    return m_params;
  }

  GCXX_FHC auto getSemaphores() const -> const deviceExternalSemaphore_t* {
    return m_params.extSemArray;
  }

  GCXX_FHC auto getSemaphoreParams() const
    -> const deviceExternalSemaphoreSignalParams_t* {
    return m_params.paramsArray;
  }

  GCXX_FHC auto getNumExtSems() const -> unsigned int {
    return m_params.numExtSems;
  }

 protected:
  GCXX_FH ExternalSemaphoreSignalNodeParamsView() = default;

  GCXX_FHC explicit ExternalSemaphoreSignalNodeParamsView(
    deviceExternalSemaphoreSignalNodeParams_t params)
      : m_params{params} {}

  // Initialized once via the constructor; never mutated afterwards.
  const deviceExternalSemaphoreSignalNodeParams_t m_params{};  // NOLINT
};

class ExternalSemaphoreSignalNodeParams
    : private details_::ExtSemStore<ExternalSemaphoreSignalNodeParamsView::
                                      deviceExternalSemaphoreSignalParams_t>,
      public ExternalSemaphoreSignalNodeParamsView {
 public:
  ExternalSemaphoreSignalNodeParams() = default;

  ExternalSemaphoreSignalNodeParams(
    gcxx::span<const deviceExternalSemaphore_t> sems,
    gcxx::span<const deviceExternalSemaphoreSignalParams_t> semaphoreParams)
      : ExtSemStore{sems.data(), sems.data() + sems.size(),
                    semaphoreParams.data(),
                    semaphoreParams.data() + semaphoreParams.size()},
        ExternalSemaphoreSignalNodeParamsView{
          make_raw_params(ExtSemStore::sems.data(), ExtSemStore::params.data(),
                          ExtSemStore::sems.size())} {
    GCXX_RUNTIME_EXPECT(
      sems.size() == semaphoreParams.size(),
      "ExternalSemaphoreSignalNodeParams: semaphore and parameter array "
      "sizes must match");
  }

  // Non-copyable/non-movable so the raw array pointers cannot dangle.
  ExternalSemaphoreSignalNodeParams(const ExternalSemaphoreSignalNodeParams&) =
    delete;
  ExternalSemaphoreSignalNodeParams(ExternalSemaphoreSignalNodeParams&&) =
    delete;

  auto operator=(const ExternalSemaphoreSignalNodeParams&)
    -> ExternalSemaphoreSignalNodeParams& = delete;
  auto operator=(ExternalSemaphoreSignalNodeParams&&)
    -> ExternalSemaphoreSignalNodeParams& = delete;

  ~ExternalSemaphoreSignalNodeParams() = default;

 private:
  using ExtSemStore =
    details_::ExtSemStore<ExternalSemaphoreSignalNodeParamsView::
                            deviceExternalSemaphoreSignalParams_t>;

  GCXX_FHC static auto make_raw_params(
    deviceExternalSemaphore_t* extSemArray,
    const deviceExternalSemaphoreSignalParams_t* paramsArray,
    std::size_t numExtSems) -> deviceExternalSemaphoreSignalNodeParams_t {
    deviceExternalSemaphoreSignalNodeParams_t p{};
    p.extSemArray = extSemArray;
    p.paramsArray = paramsArray;
    p.numExtSems  = static_cast<unsigned int>(numExtSems);
    return p;
  }
};

class ExternalSemaphoreWaitNodeParamsView {
 public:
  using deviceExternalSemaphoreWaitNodeParams_t =
    driver::deviceExternalSemaphoreWaitNodeParams_t;
  using deviceExternalSemaphore_t = driver::deviceExternalSemaphore_t;
  using deviceExternalSemaphoreWaitParams_t =
    driver::deviceExternalSemaphoreWaitParams_t;

  GCXX_FHC auto getRawParams() const
    -> const deviceExternalSemaphoreWaitNodeParams_t& {
    return m_params;
  }

  GCXX_FHC auto getSemaphores() const -> const deviceExternalSemaphore_t* {
    return m_params.extSemArray;
  }

  GCXX_FHC auto getSemaphoreParams() const
    -> const deviceExternalSemaphoreWaitParams_t* {
    return m_params.paramsArray;
  }

  GCXX_FHC auto getNumExtSems() const -> unsigned int {
    return m_params.numExtSems;
  }

 protected:
  GCXX_FH ExternalSemaphoreWaitNodeParamsView() = default;

  GCXX_FHC explicit ExternalSemaphoreWaitNodeParamsView(
    deviceExternalSemaphoreWaitNodeParams_t params)
      : m_params{params} {}

  // Initialized once via the constructor; never mutated afterwards.
  const deviceExternalSemaphoreWaitNodeParams_t m_params{};  // NOLINT
};

class ExternalSemaphoreWaitNodeParams
    : private details_::ExtSemStore<ExternalSemaphoreWaitNodeParamsView::
                                      deviceExternalSemaphoreWaitParams_t>,
      public ExternalSemaphoreWaitNodeParamsView {
 public:
  ExternalSemaphoreWaitNodeParams() = default;

  ExternalSemaphoreWaitNodeParams(
    gcxx::span<const deviceExternalSemaphore_t> sems,
    gcxx::span<const deviceExternalSemaphoreWaitParams_t> semaphoreParams)
      : ExtSemStore{sems.data(), sems.data() + sems.size(),
                    semaphoreParams.data(),
                    semaphoreParams.data() + semaphoreParams.size()},
        ExternalSemaphoreWaitNodeParamsView{
          make_raw_params(ExtSemStore::sems.data(), ExtSemStore::params.data(),
                          ExtSemStore::sems.size())} {
    GCXX_RUNTIME_EXPECT(
      sems.size() == semaphoreParams.size(),
      "ExternalSemaphoreWaitNodeParams: semaphore and parameter array "
      "sizes must match");
  }

  // Non-copyable/non-movable so the raw array pointers cannot dangle.
  ExternalSemaphoreWaitNodeParams(const ExternalSemaphoreWaitNodeParams&) =
    delete;
  ExternalSemaphoreWaitNodeParams(ExternalSemaphoreWaitNodeParams&&) = delete;

  auto operator=(const ExternalSemaphoreWaitNodeParams&)
    -> ExternalSemaphoreWaitNodeParams& = delete;
  auto operator=(ExternalSemaphoreWaitNodeParams&&)
    -> ExternalSemaphoreWaitNodeParams& = delete;

  ~ExternalSemaphoreWaitNodeParams() = default;

 private:
  using ExtSemStore = details_::ExtSemStore<
    ExternalSemaphoreWaitNodeParamsView::deviceExternalSemaphoreWaitParams_t>;

  GCXX_FHC static auto make_raw_params(
    deviceExternalSemaphore_t* extSemArray,
    const deviceExternalSemaphoreWaitParams_t* paramsArray,
    std::size_t numExtSems) -> deviceExternalSemaphoreWaitNodeParams_t {
    deviceExternalSemaphoreWaitNodeParams_t p{};
    p.extSemArray = extSemArray;
    p.paramsArray = paramsArray;
    p.numExtSems  = static_cast<unsigned int>(numExtSems);
    return p;
  }
};

GCXX_NAMESPACE_DETAILS_BEGIN()

namespace ext_sem_node_builder {

  struct sems_tag {};
  struct signal_params_tag {};
  struct wait_params_tag {};

}  // namespace ext_sem_node_builder

namespace esnb = ext_sem_node_builder;

template <typename... Set>
class ExternalSemaphoreSignalParamsBuilder {
 public:
  ExternalSemaphoreSignalParamsBuilder() = default;

  GCXX_FH auto setSemaphores(
    gcxx::span<
      const ExternalSemaphoreSignalNodeParamsView::deviceExternalSemaphore_t>
      sems) const
    -> ExternalSemaphoreSignalParamsBuilder<Set..., esnb::sems_tag> {
    static_assert(!details_::contains_v<esnb::sems_tag, Set...>,
                  "setSemaphores() may only be called once");
    ExternalSemaphoreSignalParamsBuilder<Set..., esnb::sems_tag> next = *this;
    next.m_sems.assign(sems.data(), sems.data() + sems.size());
    return next;
  }

  GCXX_FH auto setSignalParams(
    gcxx::span<const ExternalSemaphoreSignalNodeParamsView::
                 deviceExternalSemaphoreSignalParams_t>
      semaphoreParams) const
    -> ExternalSemaphoreSignalParamsBuilder<Set..., esnb::signal_params_tag> {
    static_assert(!details_::contains_v<esnb::signal_params_tag, Set...>,
                  "setSignalParams() may only be called once");
    ExternalSemaphoreSignalParamsBuilder<Set..., esnb::signal_params_tag> next =
      *this;
    next.m_params.assign(semaphoreParams.data(),
                         semaphoreParams.data() + semaphoreParams.size());
    return next;
  }

  GCXX_FH auto build() const -> gcxx::ExternalSemaphoreSignalNodeParams {
    static_assert(details_::contains_v<esnb::sems_tag, Set...>,
                  "setSemaphores() required before build()");
    static_assert(details_::contains_v<esnb::signal_params_tag, Set...>,
                  "setSignalParams() required before build()");
    return gcxx::ExternalSemaphoreSignalNodeParams{
      gcxx::span<
        const ExternalSemaphoreSignalNodeParamsView::deviceExternalSemaphore_t>{
        m_sems.data(), m_sems.size()},
      gcxx::span<const ExternalSemaphoreSignalNodeParamsView::
                   deviceExternalSemaphoreSignalParams_t>{m_params.data(),
                                                          m_params.size()}};
  }

 private:
  template <typename...>
  friend class ExternalSemaphoreSignalParamsBuilder;  // states construct each
                                                      // other

  // State hand-off between builder states.
  template <typename... Other>
  ExternalSemaphoreSignalParamsBuilder(
    const ExternalSemaphoreSignalParamsBuilder<Other...>& other)
      : m_sems{other.m_sems}, m_params{other.m_params} {}

  std::vector<ExternalSemaphoreSignalNodeParamsView::deviceExternalSemaphore_t>
    m_sems{};
  std::vector<ExternalSemaphoreSignalNodeParamsView::
                deviceExternalSemaphoreSignalParams_t>
    m_params{};
};

template <typename... Set>
class ExternalSemaphoreWaitParamsBuilder {
 public:
  ExternalSemaphoreWaitParamsBuilder() = default;

  GCXX_FH auto setSemaphores(
    gcxx::span<
      const ExternalSemaphoreWaitNodeParamsView::deviceExternalSemaphore_t>
      sems) const
    -> ExternalSemaphoreWaitParamsBuilder<Set..., esnb::sems_tag> {
    static_assert(!details_::contains_v<esnb::sems_tag, Set...>,
                  "setSemaphores() may only be called once");
    ExternalSemaphoreWaitParamsBuilder<Set..., esnb::sems_tag> next = *this;
    next.m_sems.assign(sems.data(), sems.data() + sems.size());
    return next;
  }

  GCXX_FH auto setWaitParams(
    gcxx::span<const ExternalSemaphoreWaitNodeParamsView::
                 deviceExternalSemaphoreWaitParams_t>
      semaphoreParams) const
    -> ExternalSemaphoreWaitParamsBuilder<Set..., esnb::wait_params_tag> {
    static_assert(!details_::contains_v<esnb::wait_params_tag, Set...>,
                  "setWaitParams() may only be called once");
    ExternalSemaphoreWaitParamsBuilder<Set..., esnb::wait_params_tag> next =
      *this;
    next.m_params.assign(semaphoreParams.data(),
                         semaphoreParams.data() + semaphoreParams.size());
    return next;
  }

  GCXX_FH auto build() const -> gcxx::ExternalSemaphoreWaitNodeParams {
    static_assert(details_::contains_v<esnb::sems_tag, Set...>,
                  "setSemaphores() required before build()");
    static_assert(details_::contains_v<esnb::wait_params_tag, Set...>,
                  "setWaitParams() required before build()");
    return gcxx::ExternalSemaphoreWaitNodeParams{
      gcxx::span<
        const ExternalSemaphoreWaitNodeParamsView::deviceExternalSemaphore_t>{
        m_sems.data(), m_sems.size()},
      gcxx::span<const ExternalSemaphoreWaitNodeParamsView::
                   deviceExternalSemaphoreWaitParams_t>{m_params.data(),
                                                        m_params.size()}};
  }

 private:
  template <typename...>
  friend class ExternalSemaphoreWaitParamsBuilder;  // states construct each
                                                    // other

  // State hand-off between builder states.
  template <typename... Other>
  ExternalSemaphoreWaitParamsBuilder(
    const ExternalSemaphoreWaitParamsBuilder<Other...>& other)
      : m_sems{other.m_sems}, m_params{other.m_params} {}

  std::vector<ExternalSemaphoreWaitNodeParamsView::deviceExternalSemaphore_t>
    m_sems{};
  std::vector<
    ExternalSemaphoreWaitNodeParamsView::deviceExternalSemaphoreWaitParams_t>
    m_params{};
};

GCXX_NAMESPACE_DETAILS_END()

using ExternalSemaphoreSignalParamsBuilder_t =
  details_::ExternalSemaphoreSignalParamsBuilder<>;

GCXX_FH auto ExternalSemaphoreSignalParamsBuilder()
  -> ExternalSemaphoreSignalParamsBuilder_t {
  return {};
}

using ExternalSemaphoreWaitParamsBuilder_t =
  details_::ExternalSemaphoreWaitParamsBuilder<>;

GCXX_FH auto ExternalSemaphoreWaitParamsBuilder()
  -> ExternalSemaphoreWaitParamsBuilder_t {
  return {};
}

GCXX_NAMESPACE_MAIN_END()


#endif
