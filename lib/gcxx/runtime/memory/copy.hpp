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

// Copy via smart/raw pointers.
GCXX_TEMPLATE(typename Ptr1, typename Ptr2)
GCXX_REQUIRES(details_::is_pointer_or_has_get_v<Ptr1> GCXX_AND
                details_::is_pointer_or_has_get_v<Ptr2>)
GCXX_FH auto Copy(Ptr1&& destination, Ptr2&& source,
                  const std::size_t numElements) -> void {
  using VT1 = typename details_::pointed_to_type_t<Ptr1>;
  using VT2 = typename details_::pointed_to_type_t<Ptr2>;
  static_assert(std::is_same_v<VT1, VT2>,
                "copy needs pointers to point to same type");
  driver::deviceCopy(details_::get_raw_pointer(destination),
                     details_::get_raw_pointer(source),
                     numElements * sizeof(VT1));
}

GCXX_TEMPLATE(typename Ptr1, typename Ptr2)
GCXX_REQUIRES(details_::is_pointer_or_has_get_v<Ptr1> GCXX_AND
                details_::is_pointer_or_has_get_v<Ptr2>)
GCXX_FH auto Copy(const StreamView& stream, Ptr1&& destination, Ptr2&& source,
                  const std::size_t numElements) -> void {
  using VT1 = typename details_::pointed_to_type_t<Ptr1>;
  using VT2 = typename details_::pointed_to_type_t<Ptr2>;
  static_assert(std::is_same_v<VT1, VT2>,
                "copy needs pointers to point to same type");
  driver::deviceCopyAsync(details_::get_raw_pointer(destination),
                          details_::get_raw_pointer(source),
                          numElements * sizeof(VT1), stream.getRawHandle());
}

// Copy between span-like types.
GCXX_TEMPLATE(typename DSTTY, typename SRCTY)
GCXX_REQUIRES(is_span_like_v<DSTTY> GCXX_AND is_span_like_v<SRCTY>)
GCXX_FH auto Copy(DSTTY&& destination, SRCTY&& source) -> void {
  static_assert(std::is_same_v<span_element_t<DSTTY>, span_element_t<SRCTY>>,
                "copy needs spans like data struct to point to same type");
  driver::deviceCopy(
    details_::to_address(details_::data(destination)),
    details_::to_address(details_::data(source)),
    details_::size(destination) * sizeof(span_element_t<DSTTY>));
}

GCXX_TEMPLATE(typename DSTTY, typename SRCTY)
GCXX_REQUIRES(is_span_like_v<DSTTY> GCXX_AND is_span_like_v<SRCTY>)
GCXX_FH auto Copy(const StreamView& stream, DSTTY&& destination,
                  SRCTY&& source) -> void {
  static_assert(std::is_same_v<span_element_t<DSTTY>, span_element_t<SRCTY>>,
                "copy needs spans like data struct to point to same type");
  driver::deviceCopyAsync(
    details_::to_address(details_::data(destination)),
    details_::to_address(details_::data(source)),
    details_::size(destination) * sizeof(span_element_t<DSTTY>),
    stream.getRawHandle());
}

GCXX_NAMESPACE_MAIN_END()


#endif
