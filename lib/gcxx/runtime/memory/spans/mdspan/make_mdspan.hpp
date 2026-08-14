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

// length = number of strided elements reachable in the storage (stride >= 1):
// floor((size-1)/stride)+1, or 0 for empty storage. Shared by make_vector and
// make_device_vector.
template <class Idx, class SpanLike>
constexpr auto strided_length(const SpanLike& storage, Idx stride) -> Idx {
  const Idx cap = static_cast<Idx>(gcxx::details_::size(storage));
  return (cap > Idx{0}) ? ((cap - Idx{1}) / stride) + Idx{1} : Idx{0};
}

GCXX_NAMESPACE_MAIN_DETAILS_END()

GCXX_NAMESPACE_MAIN_BEGIN()

// build rank 1 mdspan for use with Level 1&2 blas using a linear data
// structure like vector, buffer or spans
//
// The index_type Idx defaults to int; override it explicitly with the first
// template argument (e.g. make_vector<int64_t>(...) for the *_64 BLAS path),
// or let it be deduced from a passed `stride`.
//
// Example:
// auto x = make_vector(gcxx::span(dX.get(), cap));      // contiguous, int
// auto y = make_vector(gcxx::span(buf, cap), 2);        // every other, int
// auto z = make_vector<int64_t>(gcxx::span(buf));       // contiguous, int64
//
// The vector length is *derived* from the storage: how many elements are
// reachable stepping by `stride`, floor((size-1)/stride)+1. Deriving it (rather
// than taking n = size) keeps the view in-bounds for stride > 1 — a span of 10
// elements at stride 3 yields a length-4 vector (indices 0,3,6,9).
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

// Same as make_vector, but the returned view is marked device-resident via
// gcxx::device_accessor — the form the gcxx::blas operations require for
// vector operands.
//
// Example:
// auto x = make_device_vector(gcxx::span(dX.get(), cap));
GCXX_TEMPLATE(class Idx = int, class SpanLike)
GCXX_REQUIRES(gcxx::is_span_like_v<SpanLike> GCXX_AND std::is_integral_v<Idx>)
constexpr auto make_device_vector(SpanLike&& storage, Idx stride = Idx{1})
  -> gcxx::mdspan<gcxx::span_element_t<SpanLike>,
                  gcxx::extents<Idx, gcxx::dynamic_extent>,
                  gcxx::layout_stride,
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
