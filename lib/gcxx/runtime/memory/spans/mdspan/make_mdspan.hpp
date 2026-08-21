// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_SPANS_MDSPAN_MAKE_MDSPAN_HPP_
#define GCXX_RUNTIME_MEMORY_SPANS_MDSPAN_MAKE_MDSPAN_HPP_

#include <array>
#include <cstddef>
#include <type_traits>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/memory/spans/mdspan/mdspan.hpp>
#include <gcxx/runtime/memory/spans/span/span.hpp>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN()

// Reachable elements at stride >= 1: floor((size-1)/stride)+1, else 0.
template <class Idx, class SpanLike>
constexpr auto strided_length(const SpanLike& storage, Idx stride) -> Idx {
  const Idx cap = static_cast<Idx>(gcxx::details_::size(storage));
  return (cap > Idx{0}) ? ((cap - Idx{1}) / stride) + Idx{1} : Idx{0};
}

GCXX_NAMESPACE_MAIN_DETAILS_END()

GCXX_NAMESPACE_MAIN_BEGIN()

// Length derived from stride (stays in-bounds); Idx defaults to int.
GCXX_TEMPLATE(class Idx = int, class SpanLike)
GCXX_REQUIRES(gcxx::is_span_like_v<SpanLike> GCXX_AND std::is_integral_v<Idx>)
constexpr auto make_vector(SpanLike&& storage, Idx stride = Idx{1})
  -> gcxx::mdspan<gcxx::span_element_t<SpanLike>,
                  gcxx::extents<Idx, gcxx::dynamic_extent>,
                  gcxx::layout_stride> {
  using T         = gcxx::span_element_t<SpanLike>;
  using extents_t = gcxx::extents<Idx, gcxx::dynamic_extent>;

  const Idx n = gcxx::details_::strided_length<Idx>(storage, stride);

  const gcxx::layout_stride::mapping<extents_t> map(extents_t{n},
                                                    std::array{stride});
  return gcxx::mdspan<T, extents_t, gcxx::layout_stride>(
    gcxx::details_::data(storage), map);
}

// make_vector with a device_accessor — what gcxx::blas requires.
GCXX_TEMPLATE(class Idx = int, class SpanLike)
GCXX_REQUIRES(gcxx::is_span_like_v<SpanLike> GCXX_AND std::is_integral_v<Idx>)
constexpr auto make_device_vector(SpanLike&& storage, Idx stride = Idx{1})
  -> gcxx::mdspan<gcxx::span_element_t<SpanLike>,
                  gcxx::extents<Idx, gcxx::dynamic_extent>, gcxx::layout_stride,
                  gcxx::device_accessor<
                    gcxx::default_accessor<gcxx::span_element_t<SpanLike>>>> {
  using T          = gcxx::span_element_t<SpanLike>;
  using extents_t  = gcxx::extents<Idx, gcxx::dynamic_extent>;
  using accessor_t = gcxx::device_accessor<gcxx::default_accessor<T>>;

  const Idx n = gcxx::details_::strided_length<Idx>(storage, stride);

  const gcxx::layout_stride::mapping<extents_t> map(extents_t{n},
                                                    std::array{stride});
  return gcxx::mdspan<T, extents_t, gcxx::layout_stride, accessor_t>(
    gcxx::details_::data(storage), map);
}

GCXX_NAMESPACE_MAIN_END()

#endif
