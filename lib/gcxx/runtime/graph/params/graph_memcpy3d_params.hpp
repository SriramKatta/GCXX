// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_MEMCPY3D_PARAMS_HPP_
#define GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_MEMCPY3D_PARAMS_HPP_

#include <cstddef>
#include <utility>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime_backend/backend_graph.hpp>

#include <gcxx/runtime/memory/memory_helpers.hpp>

// Type-state builder: setSrcPtr(), setDstPtr() and setExtent() are required
// and may be called only once; build() refuses to compile otherwise.
// setSrcPos()/setDstPos() are optional (default {0, 0, 0}).

GCXX_NAMESPACE_MAIN_BEGIN()

class Memcpy3DParamsView {
 public:
  using deviceMemcpy3DParams_t = driver::deviceMemcpy3DParams_t;

  GCXX_FHC auto getRawParams() const -> const deviceMemcpy3DParams_t& {
    return m_params;
  }

  GCXX_FHC auto getSrcPos() const -> gcxx::devicePos { return m_params.srcPos; }

  GCXX_FHC auto getDstPos() const -> gcxx::devicePos { return m_params.dstPos; }

  GCXX_FHC auto getSrcPtr() const -> const gcxx::devicePitchedPtr {
    return m_params.srcPtr;
  }

  GCXX_FHC auto getDstPtr() const -> const gcxx::devicePitchedPtr {
    return m_params.dstPtr;
  }

  GCXX_FHC auto getExtent() const -> const gcxx::deviceExtent {
    return m_params.extent;
  }

 protected:
  GCXX_FH Memcpy3DParamsView() = default;

  GCXX_FHC explicit Memcpy3DParamsView(deviceMemcpy3DParams_t params)
      : m_params{params} {}

  // Initialized once via the constructor; never mutated afterwards.
  deviceMemcpy3DParams_t m_params{};  // NOLINT
};

// Adds no state over the View; kept only for uniformity with the params
// kinds that do carry storage (kernel, mem-alloc, external-semaphore).
class Memcpy3DParams : public Memcpy3DParamsView {
 public:
  GCXX_FHC Memcpy3DParams() = default;

  GCXX_FHC Memcpy3DParams(const gcxx::devicePitchedPtr& srcPtr,
                          gcxx::devicePos srcPos,
                          const gcxx::devicePitchedPtr& dstPtr,
                          gcxx::devicePos dstPos, gcxx::deviceExtent extent)
      : Memcpy3DParamsView{
          make_raw_params(srcPtr, srcPos, dstPtr, dstPos, extent)} {}

  // Non-copyable/non-movable so pointers into m_params cannot dangle.
  Memcpy3DParams(const Memcpy3DParams&) = delete;
  Memcpy3DParams(Memcpy3DParams&&)      = delete;

  auto operator=(const Memcpy3DParams&) -> Memcpy3DParams& = delete;
  auto operator=(Memcpy3DParams&&) -> Memcpy3DParams&      = delete;

  ~Memcpy3DParams() = default;

 private:
  GCXX_FHC static auto make_raw_params(
    const gcxx::devicePitchedPtr& srcPtr, gcxx::devicePos srcPos,
    const gcxx::devicePitchedPtr& dstPtr, gcxx::devicePos dstPos,
    gcxx::deviceExtent extent) -> deviceMemcpy3DParams_t {
    deviceMemcpy3DParams_t p{};
    p.srcPtr = srcPtr;
    p.srcPos = srcPos;
    p.dstPtr = dstPtr;
    p.dstPos = dstPos;
    p.extent = extent;
    p.kind   = GCXX_RUNTIME_BACKEND(MemcpyDefault);
    return p;
  }
};

GCXX_NAMESPACE_DETAILS_BEGIN()

namespace memcpy3d_builder {

  struct srcptr_tag {};
  struct dstptr_tag {};
  struct extent_tag {};

}  // namespace memcpy3d_builder

namespace m3b = memcpy3d_builder;

template <typename... Set>
class Memcpy3DParamsBuilder {
 public:
  Memcpy3DParamsBuilder() = default;

  GCXX_FHC auto setSrcPtr(const gcxx::devicePitchedPtr& ptr) const
    -> Memcpy3DParamsBuilder<Set..., m3b::srcptr_tag> {
    static_assert(!details_::contains_v<m3b::srcptr_tag, Set...>,
                  "setSrcPtr() may only be called once");
    Memcpy3DParamsBuilder<Set..., m3b::srcptr_tag> next = *this;
    next.m_srcPtr                                       = ptr;
    return next;
  }

  GCXX_FHC auto setDstPtr(const gcxx::devicePitchedPtr& ptr) const
    -> Memcpy3DParamsBuilder<Set..., m3b::dstptr_tag> {
    static_assert(!details_::contains_v<m3b::dstptr_tag, Set...>,
                  "setDstPtr() may only be called once");
    Memcpy3DParamsBuilder<Set..., m3b::dstptr_tag> next = *this;
    next.m_dstPtr                                       = ptr;
    return next;
  }

  GCXX_FHC auto setExtent(gcxx::deviceExtent ext) const
    -> Memcpy3DParamsBuilder<Set..., m3b::extent_tag> {
    static_assert(!details_::contains_v<m3b::extent_tag, Set...>,
                  "setExtent() may only be called once");
    Memcpy3DParamsBuilder<Set..., m3b::extent_tag> next = *this;
    next.m_extent                                       = ext;
    return next;
  }

  GCXX_FHC auto setSrcPos(gcxx::devicePos pos) const -> Memcpy3DParamsBuilder {
    Memcpy3DParamsBuilder next = *this;
    next.m_srcPos              = pos;
    return next;
  }

  GCXX_FHC auto setDstPos(gcxx::devicePos pos) const -> Memcpy3DParamsBuilder {
    Memcpy3DParamsBuilder next = *this;
    next.m_dstPos              = pos;
    return next;
  }

  GCXX_FHC auto build() const -> gcxx::Memcpy3DParams {
    static_assert(details_::contains_v<m3b::srcptr_tag, Set...>,
                  "setSrcPtr() required before build()");
    static_assert(details_::contains_v<m3b::dstptr_tag, Set...>,
                  "setDstPtr() required before build()");
    static_assert(details_::contains_v<m3b::extent_tag, Set...>,
                  "setExtent() required before build()");
    return gcxx::Memcpy3DParams{m_srcPtr, m_srcPos, m_dstPtr, m_dstPos,
                                m_extent};
  }

 private:
  template <typename...>
  friend class Memcpy3DParamsBuilder;  // states construct each other

  // State hand-off between builder states.
  template <typename... Other>
  Memcpy3DParamsBuilder(const Memcpy3DParamsBuilder<Other...>& other)
      : m_srcPtr{other.m_srcPtr},
        m_srcPos{other.m_srcPos},
        m_dstPtr{other.m_dstPtr},
        m_dstPos{other.m_dstPos},
        m_extent{other.m_extent} {}

  gcxx::devicePitchedPtr m_srcPtr{};
  gcxx::devicePos m_srcPos{0, 0, 0};
  gcxx::devicePitchedPtr m_dstPtr{};
  gcxx::devicePos m_dstPos{0, 0, 0};
  gcxx::deviceExtent m_extent{};
};

GCXX_NAMESPACE_DETAILS_END()

using Memcpy3DParamsBuilder_t = details_::Memcpy3DParamsBuilder<>;

GCXX_FH auto Memcpy3DParamsBuilder() -> Memcpy3DParamsBuilder_t {
  return {};
}

GCXX_NAMESPACE_MAIN_END()


#endif
