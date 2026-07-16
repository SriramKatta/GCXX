// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_API_RUNTIME_MEMORY_MEMSET_HPP_
#define GCXX_API_RUNTIME_MEMORY_MEMSET_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime/memory/smartpointers/pointers.hpp>
#include <gcxx/runtime/memory/spans/spans.hpp>
#include <gcxx/runtime/stream.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>
#include <gcxx/runtime_backend/backend_memory.hpp>
#include <type_traits>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()

// ╔════════════════════════════════════════════════════════╗
// ║   smart / raw pointer version based on element type    ║
// ╚════════════════════════════════════════════════════════╝
GCXX_TEMPLATE(typename Ptr)
GCXX_REQUIRES(details_::is_pointer_or_has_get_v<Ptr>)
GCXX_FH auto Memset(Ptr&& handle, const int value,
                    const std::size_t numElements) -> void {
  auto raw_ptr = details_::get_raw_pointer(handle);
  using VT     = typename details_::pointed_to_type_t<Ptr>;
  driver::deviceMemset(raw_ptr, value, numElements * sizeof(VT));
}

GCXX_TEMPLATE(typename Ptr)
GCXX_REQUIRES(details_::is_pointer_or_has_get_v<Ptr>)
GCXX_FH auto Memset(const StreamView& stream, Ptr&& handle, const int value,
                    const std::size_t numElements) -> void {
  auto raw_ptr = details_::get_raw_pointer(handle);
  using VT     = typename details_::pointed_to_type_t<Ptr>;
  driver::deviceMemsetAsync(raw_ptr, value, numElements * sizeof(VT), stream);
}


// ╔════════════════════════════════════════════════════════╗
// ║  works on any type that can be converted into a span   ║
// ╚════════════════════════════════════════════════════════╝
GCXX_TEMPLATE(typename DSTTY)
GCXX_REQUIRES(is_span_like_v<DSTTY>)
GCXX_FH auto Memset(DSTTY&& destination, const int value) -> void {
  driver::deviceMemset(
    details_::to_address(details_::data(destination)), value,
    details_::size(destination) * sizeof(span_element_t<DSTTY>));
}

GCXX_TEMPLATE(typename DSTTY)
GCXX_REQUIRES(is_span_like_v<DSTTY>)
GCXX_FH auto Memset(const StreamView& stream, DSTTY&& destination,
                    const int value) -> void {
  driver::deviceMemsetAsync(
    details_::to_address(details_::data(destination)), value,
    details_::size(destination) * sizeof(span_element_t<DSTTY>), stream);
}
GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()


#endif