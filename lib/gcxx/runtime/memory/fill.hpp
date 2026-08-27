// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_API_RUNTIME_MEMORY_FILL_HPP_
#define GCXX_API_RUNTIME_MEMORY_FILL_HPP_

#include <cstddef>
#include <type_traits>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime/launch.hpp>
#include <gcxx/runtime/memory/memset.hpp>
#include <gcxx/runtime/memory/smartpointers/pointers.hpp>
#include <gcxx/runtime/memory/spans/spans.hpp>
#include <gcxx/runtime/stream.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


// Only invoked for non-zero values; zero goes through the fast memset path.
template <typename VT>
__global__ void fill_kernel(VT* ptr, VT value, std::size_t n) {
  const std::size_t idx =
    static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    ptr[idx] = value;
  }
}

// Single decision point: VT{} values use byte memset, others use kernel.
template <typename VT>
GCXX_FH auto fill_dispatch(const StreamView& stream, VT* ptr, const VT& value,
                           std::size_t n) -> void {
  if (n == 0) {
    return;
  }
  if (value == VT{}) {
    Memset(stream, ptr, 0, n);
  } else {
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size =
      static_cast<unsigned int>((n + block_size - 1) / block_size);
    launch::Kernel(stream, dim3(grid_size), dim3(block_size), 0,
                   fill_kernel<VT>, ptr, static_cast<VT>(value), n);
  }
}

// Fill via smart/raw pointers.
GCXX_TEMPLATE(typename Ptr, typename Val)
GCXX_REQUIRES(details_::is_pointer_or_has_get_v<Ptr>)
GCXX_FH auto Fill(Ptr& handle, const Val& value,
                  const std::size_t numElements) -> void {
  Fill(StreamView::Null(), handle, value, numElements);
}

GCXX_TEMPLATE(typename Ptr, typename Val)
GCXX_REQUIRES(details_::is_pointer_or_has_get_v<Ptr>)
GCXX_FH auto Fill(const StreamView& stream, Ptr& handle, const Val& value,
                  const std::size_t numElements) -> void {
  using element_t = typename details_::pointed_to_type_t<Ptr>;
  fill_dispatch(stream, details_::get_raw_pointer(handle),
                static_cast<element_t>(value), numElements);
}

// Fill of span-like destinations.
GCXX_TEMPLATE(typename DSTTY, typename Val)
GCXX_REQUIRES(is_span_like_v<DSTTY>)
GCXX_FH auto Fill(DSTTY&& destination, const Val& value) -> void {
  Fill(StreamView::Null(), std::forward<DSTTY>(destination), value);
}

GCXX_TEMPLATE(typename DSTTY, typename Val)
GCXX_REQUIRES(is_span_like_v<DSTTY>)
// Inspection-only: data()/size() take lvalue refs, so forwarding an rvalue
// destination would feed a const pointer into fill_dispatch.
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
GCXX_FH auto Fill(const StreamView& stream, DSTTY&& destination,
                  const Val& value) -> void {
  using element_t = span_element_t<DSTTY>;
  fill_dispatch(stream, details_::to_address(details_::data(destination)),
                static_cast<element_t>(value), details_::size(destination));
}

GCXX_NAMESPACE_MAIN_END()

#endif
