// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_MEMSET_PARAMS_HPP_
#define GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_MEMSET_PARAMS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_graph.hpp>

#include <gcxx/runtime/memory/memory_helpers.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class MemsetParamsView {
 public:
  using deviceMemsetParams_t = driver::deviceMemsetParams_t;

  GCXX_FHC auto getRawParams() const -> const deviceMemsetParams_t& {
    return m_params;
  }

  GCXX_FHC auto getPtr() const -> const void* const { return m_params.dst; }

  GCXX_FHC auto getPitch() const -> const size_t { return m_params.pitch; }

  GCXX_FHC auto getValue() const -> const unsigned int {
    return m_params.value;
  }

  GCXX_FHC auto getElementSize() const -> const unsigned int {
    return m_params.elementSize;
  }

  GCXX_FHC auto getWidth() const -> const size_t { return m_params.width; }

  GCXX_FHC auto getHeight() const -> const size_t { return m_params.height; }

 protected:
  GCXX_FHC MemsetParamsView() { std::memset(&m_params, 0, sizeof(m_params)); }

  deviceMemsetParams_t m_params{};  // NOLINT
};

class MemsetParams : public MemsetParamsView {

 public:
  GCXX_FHC MemsetParams() = default;

  GCXX_FHC
  MemsetParams(void* dst, size_t pitch, unsigned int value,
               unsigned int elementSize, size_t width, size_t height) {
    m_params.dst         = dst;
    m_params.pitch       = pitch;
    m_params.value       = value;
    m_params.elementSize = elementSize;
    m_params.width       = width;
    m_params.height      = height;
  }

  // Disable move/copy to ensure params_ remains stable
  MemsetParams(const MemsetParams&)           = delete;
  MemsetParams operator=(const MemsetParams&) = delete;

  MemsetParams(MemsetParams&&)           = delete;
  MemsetParams operator=(MemsetParams&&) = delete;

  ~MemsetParams() = default;
};

GCXX_NAMESPACE_DETAILS_BEGIN()

class MemsetParamsBuilder {
 public:
  GCXX_FH static auto create() -> MemsetParamsBuilder { return {}; }

  GCXX_FHC auto setPtr(void* ptr) -> MemsetParamsBuilder& {
    m_dst = ptr;
    return *this;
  }

  GCXX_FHC auto setPitch(size_t pitch) -> MemsetParamsBuilder& {
    m_pitch = pitch;
    return *this;
  }

  GCXX_FHC auto setValue(unsigned int value) -> MemsetParamsBuilder& {
    m_value = value;
    return *this;
  }

  GCXX_FHC auto setElemetSize(unsigned int size) -> MemsetParamsBuilder& {
    m_elementSize = size;
    return *this;
  }

  GCXX_FHC auto setWidth(size_t width) -> MemsetParamsBuilder& {
    m_width = width;
    return *this;
  }

  GCXX_FHC auto setHeight(size_t height) -> MemsetParamsBuilder& {
    m_height = height;
    return *this;
  }

  GCXX_FHC gcxx::MemsetParams build() {
    return {m_dst, m_pitch, m_value, m_elementSize, m_width, m_height};
  }

 private:
  void* m_dst{nullptr};
  size_t m_pitch{0};
  unsigned int m_value{0};
  unsigned int m_elementSize{1};
  size_t m_width{1};
  size_t m_height{1};
};

GCXX_NAMESPACE_DETAILS_END()

// helper to simplify usage
GCXX_FH auto MemsetParamsBuilder() -> details_::MemsetParamsBuilder {
  return details_::MemsetParamsBuilder::create();
}

GCXX_NAMESPACE_MAIN_END()


#endif