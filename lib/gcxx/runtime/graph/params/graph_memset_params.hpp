// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_MEMSET_PARAMS_HPP_
#define GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_MEMSET_PARAMS_HPP_

#include <cstddef>
#include <cstring>
#include <utility>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime_backend/backend_graph.hpp>

#include <gcxx/runtime/memory/memory_helpers.hpp>

// Type-state builder: setPtr(), setValue() and setWidth() are required and
// may be called only once; build() refuses to compile otherwise. setPitch(),
// setElementSize() and setHeight() are optional (defaults 0 / 1 / 1).

GCXX_NAMESPACE_MAIN_BEGIN()

class MemsetParamsView {
 public:
  using deviceMemsetParams_t = driver::deviceMemsetParams_t;

  GCXX_FHC auto getRawParams() const -> const deviceMemsetParams_t& {
    return m_params;
  }

  GCXX_FHC auto getPtr() const -> const void* { return m_params.dst; }

  GCXX_FHC auto getPitch() const -> std::size_t { return m_params.pitch; }

  GCXX_FHC auto getValue() const -> unsigned int { return m_params.value; }

  GCXX_FHC auto getElementSize() const -> unsigned int {
    return m_params.elementSize;
  }

  GCXX_FHC auto getWidth() const -> std::size_t { return m_params.width; }

  GCXX_FHC auto getHeight() const -> std::size_t { return m_params.height; }

 protected:
  GCXX_FH MemsetParamsView() = default;

  GCXX_FHC explicit MemsetParamsView(deviceMemsetParams_t params)
      : m_params{params} {}

  // Initialized once via the constructor; never mutated afterwards.
  const deviceMemsetParams_t m_params{};  // NOLINT
};

class MemsetParams : public MemsetParamsView {
 public:
  GCXX_FHC MemsetParams() = default;

  GCXX_FHC MemsetParams(void* dst, std::size_t pitch, unsigned int value,
                        unsigned int elementSize, std::size_t width,
                        std::size_t height)
      : MemsetParamsView{
          make_raw_params(dst, pitch, value, elementSize, width, height)} {}

  // Non-copyable/non-movable so pointers into m_params cannot dangle.
  MemsetParams(const MemsetParams&)                    = delete;
  auto operator=(const MemsetParams&) -> MemsetParams& = delete;

  MemsetParams(MemsetParams&&)                    = delete;
  auto operator=(MemsetParams&&) -> MemsetParams& = delete;

  ~MemsetParams() = default;

 private:
  GCXX_FHC static auto make_raw_params(
    void* dst, std::size_t pitch, unsigned int value, unsigned int elementSize,
    std::size_t width, std::size_t height) -> deviceMemsetParams_t {
    deviceMemsetParams_t p{};
    p.dst         = dst;
    p.pitch       = pitch;
    p.value       = value;
    p.elementSize = elementSize;
    p.width       = width;
    p.height      = height;
    return p;
  }
};

GCXX_NAMESPACE_DETAILS_BEGIN()

namespace memset_builder {

  struct dst_tag {};
  struct value_tag {};
  struct width_tag {};

  // cudaMemsetParams::elementSize must be 1, 2, or 4 bytes.
  template <auto Bytes>
  inline constexpr bool is_valid_element_size_v =
    (Bytes == 1 || Bytes == 2 || Bytes == 4);

}  // namespace memset_builder

namespace msb = memset_builder;

template <typename... Set>
class MemsetParamsBuilder {
 public:
  MemsetParamsBuilder() = default;

  GCXX_FHC auto setPtr(void* ptr) const
    -> MemsetParamsBuilder<Set..., msb::dst_tag> {
    static_assert(!details_::contains_v<msb::dst_tag, Set...>,
                  "setPtr() may only be called once");
    MemsetParamsBuilder<Set..., msb::dst_tag> next = *this;
    next.m_dst                                     = ptr;
    return next;
  }

  GCXX_FHC auto setValue(unsigned int value) const
    -> MemsetParamsBuilder<Set..., msb::value_tag> {
    static_assert(!details_::contains_v<msb::value_tag, Set...>,
                  "setValue() may only be called once");
    MemsetParamsBuilder<Set..., msb::value_tag> next = *this;
    next.m_value                                     = value;
    return next;
  }

  GCXX_FHC auto setWidth(std::size_t width) const
    -> MemsetParamsBuilder<Set..., msb::width_tag> {
    static_assert(!details_::contains_v<msb::width_tag, Set...>,
                  "setWidth() may only be called once");
    MemsetParamsBuilder<Set..., msb::width_tag> next = *this;
    next.m_width                                     = width;
    return next;
  }

  GCXX_FHC auto setPitch(std::size_t pitch) const -> MemsetParamsBuilder {
    MemsetParamsBuilder next = *this;
    next.m_pitch             = pitch;
    return next;
  }

  // Element size taken from the type.
  template <typename VT>
  GCXX_FHC auto setElementSize() const -> MemsetParamsBuilder {
    static_assert(
      msb::is_valid_element_size_v<sizeof(VT)>,
      "cudaMemsetParams elementSize must be 1, 2, or 4 bytes (sizeof(VT) is "
      "not)");
    return setElementSizeBytes<sizeof(VT)>();
  }

  // Raw byte count as a compile-time constant.
  template <unsigned int Bytes>
  GCXX_FHC auto setElementSizeBytes() const -> MemsetParamsBuilder {
    static_assert(msb::is_valid_element_size_v<Bytes>,
                  "cudaMemsetParams elementSize must be 1, 2, or 4 bytes");
    MemsetParamsBuilder next = *this;
    next.m_elementSize       = Bytes;
    return next;
  }

  GCXX_FHC auto setHeight(std::size_t height) const -> MemsetParamsBuilder {
    MemsetParamsBuilder next = *this;
    next.m_height            = height;
    return next;
  }

  GCXX_FHC auto build() const -> gcxx::MemsetParams {
    static_assert(details_::contains_v<msb::dst_tag, Set...>,
                  "setPtr() required before build()");
    static_assert(details_::contains_v<msb::value_tag, Set...>,
                  "setValue() required before build()");
    static_assert(details_::contains_v<msb::width_tag, Set...>,
                  "setWidth() required before build()");
    return gcxx::MemsetParams{m_dst,         m_pitch, m_value,
                              m_elementSize, m_width, m_height};
  }

 private:
  template <typename...>
  friend class MemsetParamsBuilder;  // states construct each other

  // State hand-off between builder states.
  template <typename... Other>
  MemsetParamsBuilder(const MemsetParamsBuilder<Other...>& other)
      : m_dst{other.m_dst},
        m_pitch{other.m_pitch},
        m_value{other.m_value},
        m_elementSize{other.m_elementSize},
        m_width{other.m_width},
        m_height{other.m_height} {}

  void* m_dst{nullptr};
  std::size_t m_pitch{0};
  unsigned int m_value{0};
  unsigned int m_elementSize{1};
  std::size_t m_width{1};
  std::size_t m_height{1};
};

GCXX_NAMESPACE_DETAILS_END()

GCXX_FH auto MemsetParamsBuilder() -> details_::MemsetParamsBuilder<> {
  return {};
}

GCXX_NAMESPACE_MAIN_END()


#endif
