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


// ─────────────────────────────────────────────────────────────────────────────
// fill_kernel: writes value into every element of ptr[0..n). Only invoked for
// non-zero values; the zero case is handled by the (much faster) byte-level
// memset path in fill_dispatch below. Defined flat in the memory namespace
// (matching memset.hpp / copy.hpp) — a nested details_ namespace here would
// shadow gcxx::v1::details_ for the rest of the memory namespace.
// ─────────────────────────────────────────────────────────────────────────────
template <typename VT>
__global__ void fill_kernel(VT* ptr, VT value, std::size_t n) {
  const std::size_t idx =
    static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    ptr[idx] = value;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// fill_dispatch: the single point that decides memset vs kernel.
//   value == VT{}  -> memset (byte fill, driver fast path)
//   value != VT{}  -> fill_kernel (typed write per element)
// ─────────────────────────────────────────────────────────────────────────────
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

// ╔════════════════════════════════════════════════════════╗
// ║   smart / raw pointer version based on element type    ║
// ╚════════════════════════════════════════════════════════╝
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

// ╔════════════════════════════════════════════════════════╗
// ║  works on any type that can be converted into a span   ║
// ╚════════════════════════════════════════════════════════╝
GCXX_TEMPLATE(typename DSTTY, typename Val)
GCXX_REQUIRES(is_span_like_v<DSTTY>)
GCXX_FH auto Fill(DSTTY&& destination, const Val& value) -> void {
  Fill(StreamView::Null(), std::forward<DSTTY>(destination), value);
}

GCXX_TEMPLATE(typename DSTTY, typename Val)
GCXX_REQUIRES(is_span_like_v<DSTTY>)
GCXX_FH auto Fill(const StreamView& stream, DSTTY&& destination,
                  const Val& value) -> void {
  using element_t = span_element_t<DSTTY>;
  fill_dispatch(stream, details_::to_address(details_::data(destination)),
                static_cast<element_t>(value), details_::size(destination));
}

GCXX_NAMESPACE_MAIN_END()

#endif
