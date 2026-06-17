// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_MEMCPY3D_PARAMS_HPP_
#define GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_MEMCPY3D_PARAMS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_graph.hpp>

#include <gcxx/runtime/memory/memory_helpers.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class Memcpy3DParamsView {
 public:
  using deviceMemcpy3DParams_t = driver::deviceMemcpy3DParams_t;

  GCXX_FHC auto getRawParams() const -> const deviceMemcpy3DParams_t& {
    return m_params;
  }

  GCXX_FHC auto getSrcPos() const -> const gcxx::memory::devicePos {
    return m_params.srcPos;
  }

  GCXX_FHC auto getDstPos() const -> const gcxx::memory::devicePos {
    return m_params.dstPos;
  }

  GCXX_FHC auto getSrcPtr() const -> const gcxx::memory::devicePitchedPtr {
    return m_params.srcPtr;
  }

  GCXX_FHC auto getDstPtr() const -> const gcxx::memory::devicePitchedPtr {
    return m_params.dstPtr;
  }

  GCXX_FHC auto getExtent() const -> const gcxx::memory::deviceExtent {
    return m_params.extent;
  }

 protected:
  GCXX_FHC Memcpy3DParamsView() { memset(&m_params, 0, sizeof(m_params)); }

  deviceMemcpy3DParams_t m_params{};  // NOLINT
};

class Memcpy3DParams : public Memcpy3DParamsView {

 public:
  GCXX_FHC Memcpy3DParams() = default;

  GCXX_FHC
  Memcpy3DParams(const gcxx::memory::devicePitchedPtr& srcPtr,
                 gcxx::memory::devicePos srcPos,
                 const gcxx::memory::devicePitchedPtr& dstPtr,
                 gcxx::memory::devicePos dstPos,
                 gcxx::memory::deviceExtent extent) {
    m_params.srcPtr = srcPtr;
    m_params.srcPos = srcPos;
    m_params.dstPtr = dstPtr;
    m_params.dstPos = dstPos;
    m_params.extent = extent;
    m_params.kind   = GCXX_RUNTIME_BACKEND(MemcpyDefault);
  }

  // Disable move/copy to ensure params_ remains stable
  Memcpy3DParams(const Memcpy3DParams&) = delete;
  Memcpy3DParams(Memcpy3DParams&&)      = delete;

  Memcpy3DParams operator=(const Memcpy3DParams&) = delete;
  Memcpy3DParams operator=(Memcpy3DParams&&)      = delete;
  ~Memcpy3DParams()                               = default;
};

GCXX_NAMESPACE_DETAILS_BEGIN()

class Memcpy3DParamsBuilder {
 public:
  GCXX_FH static auto create() -> Memcpy3DParamsBuilder { return {}; }

  GCXX_FHC
  auto setSrcPtr(const gcxx::memory::devicePitchedPtr& ptr)
    -> Memcpy3DParamsBuilder& {
    m_srcPtr = ptr;
    return *this;
  }

  GCXX_FHC
  auto setSrcPos(gcxx::memory::devicePos pos) -> Memcpy3DParamsBuilder& {
    m_srcPos = pos;
    return *this;
  }

  GCXX_FHC
  auto setDstPtr(const gcxx::memory::devicePitchedPtr& ptr)
    -> Memcpy3DParamsBuilder& {
    m_dstPtr = ptr;
    return *this;
  }

  GCXX_FHC
  auto setDstPos(gcxx::memory::devicePos pos) -> Memcpy3DParamsBuilder& {
    m_dstPos = pos;
    return *this;
  }

  GCXX_FHC
  auto setExtent(gcxx::memory::deviceExtent ext) -> Memcpy3DParamsBuilder& {
    m_extent = ext;
    return *this;
  }

  GCXX_FHC gcxx::Memcpy3DParams build() {
    return {m_srcPtr, m_srcPos, m_dstPtr, m_dstPos, m_extent};
  }

 private:
  gcxx::memory::devicePitchedPtr m_srcPtr{};
  gcxx::memory::devicePos m_srcPos{0, 0, 0};
  gcxx::memory::devicePitchedPtr m_dstPtr{};
  gcxx::memory::devicePos m_dstPos{0, 0, 0};
  gcxx::memory::deviceExtent m_extent{};
};

GCXX_NAMESPACE_DETAILS_END()

// helper to simplify usage
GCXX_FH auto Memcpy3DParamsBuilder() -> details_::Memcpy3DParamsBuilder {
  return details_::Memcpy3DParamsBuilder::create();
}

GCXX_NAMESPACE_MAIN_END()


#endif