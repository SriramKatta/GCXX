// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_API_RUNTIME_MEMORY_COPY_HPP_
#define GCXX_API_RUNTIME_MEMORY_COPY_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/macros/template_helper_macros.hpp>
#include <gcxx/runtime/details/helper_function.hpp>
#include <gcxx/runtime/memory/smartpointers/pointers.hpp>
#include <gcxx/runtime/memory/spans/spans.hpp>
#include <gcxx/runtime/stream.hpp>
#include <gcxx/runtime_backend/backend_memory.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()
// ╔════════════════════════════════════════════════════════╗
// ║    smart / raw pointer version based on element type   ║
// ╚════════════════════════════════════════════════════════╝
GCXX_TEMPLATE(typename Ptr)
GCXX_REQUIRES(details_::is_pointer_or_has_get_v<Ptr>)
GCXX_FH auto Copy(Ptr destination, const Ptr source,
                  const std::size_t numElements) -> void {
  auto src_raw_ptr = details_::get_raw_pointer(source);
  auto dst_raw_ptr = details_::get_raw_pointer(destination);
  using VT         = typename details_::pointed_to_type_t<Ptr>;
  driver::deviceCopy(dst_raw_ptr, src_raw_ptr, numElements * sizeof(VT));
}

GCXX_TEMPLATE(typename Ptr)
GCXX_REQUIRES(details_::is_pointer_or_has_get_v<Ptr>)
GCXX_FH auto Copy(const StreamView& stream, Ptr destination, const Ptr source,
                  const std::size_t numElements) -> void {
  auto src_raw_ptr = details_::get_raw_pointer(source);
  auto dst_raw_ptr = details_::get_raw_pointer(destination);
  using VT         = typename details_::pointed_to_type_t<Ptr>;
  driver::deviceCopyAsync(dst_raw_ptr, src_raw_ptr, numElements * sizeof(VT),
                          stream);
}

// ╔════════════════════════════════════════════════════════╗
// ║  works on any type that can be converted into a span   ║
// ╚════════════════════════════════════════════════════════╝
GCXX_TEMPLATE(typename DSTTY, typename SRCTY)
GCXX_REQUIRES(is_span_like_v<DSTTY> GCXX_AND is_span_like_v<SRCTY>)
GCXX_FH auto Copy(DSTTY&& destination, SRCTY&& source) -> void {
  driver::deviceCopy(
    details_::to_address(details_::data(destination)),
    details_::to_address(details_::data(source)),
    details_::size(destination) * sizeof(span_element_t<DSTTY>));
}

GCXX_TEMPLATE(typename DSTTY, typename SRCTY)
GCXX_REQUIRES(is_span_like_v<DSTTY> GCXX_AND is_span_like_v<SRCTY>)
GCXX_FH auto Copy(const StreamView& stream, DSTTY&& destination,
                  SRCTY&& source) -> void {
  driver::deviceCopyAsync(
    details_::to_address(details_::data(destination)),
    details_::to_address(details_::data(source)),
    details_::size(destination) * sizeof(span_element_t<DSTTY>), stream);
}
GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()


#endif