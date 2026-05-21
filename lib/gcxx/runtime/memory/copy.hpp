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

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_DETAILS_BEGIN()

// ╔════════════════════════════════════════════════════════╗
// ║          works on pointer with bytes to copy           ║
// ╚════════════════════════════════════════════════════════╝

GCXX_FH auto Copy(void* destination, const void* source,
                  const std::size_t countinBytes) -> void {
  GCXX_SAFE_RUNTIME_CALL(Memcpy, "Failed to perform GPU copy", destination,
                         source, countinBytes,
                         GCXX_RUNTIME_BACKEND(MemcpyDefault));
}

GCXX_FH auto Copy(void* destination, const void* source,
                  const std::size_t countinBytes,
                  const StreamView& stream) -> void {
  GCXX_SAFE_RUNTIME_CALL(
    MemcpyAsync, "Failed to perform async GPU copy", destination, source,
    countinBytes, GCXX_RUNTIME_BACKEND(MemcpyDefault), stream.getRawStream());
}

GCXX_NAMESPACE_DETAILS_END()

namespace memory {
  // ╔════════════════════════════════════════════════════════╗
  // ║    pointer and count version based on element type     ║
  // ╚════════════════════════════════════════════════════════╝
  template <typename VT>
  GCXX_FH auto Copy(VT* destination, const VT* source,
                    const std::size_t numElements) -> void {
    details_::Copy((void*)destination, (const void*)source,
                   numElements * sizeof(VT));
  }

  template <typename VT>
  GCXX_FH auto Copy(const VT* destination, const VT* source,
                    const std::size_t numElements,
                    const StreamView& stream) -> void {
    details_::Copy((void*)destination, (const void*)source,
                   numElements * sizeof(VT), stream);
  }

  // ╔════════════════════════════════════════════════════════╗
  // ║  works on any type that can be converted into a span   ║
  // ╚════════════════════════════════════════════════════════╝
  GCXX_TEMPLATE(typename DSTTY, typename SRCTY)
  GCXX_REQUIRES(is_span_like_v<DSTTY> GCXX_AND is_span_like_v<SRCTY>)

  GCXX_FH auto Copy(DSTTY&& destination, SRCTY&& source) -> void {
    details_::Copy(details_::to_address(details_::data(destination)),
                   details_::to_address(details_::data(source)),
                   details_::size(destination) * sizeof(span_element_t<DSTTY>));
  }

  GCXX_TEMPLATE(typename DSTTY, typename SRCTY)
  GCXX_REQUIRES(is_span_like_v<DSTTY> GCXX_AND is_span_like_v<SRCTY>)

  GCXX_FH auto Copy(DSTTY&& destination, SRCTY&& source,
                    const StreamView& stream) -> void {
    details_::Copy(details_::to_address(details_::data(destination)),
                   details_::to_address(details_::data(source)),
                   details_::size(destination) * sizeof(span_element_t<DSTTY>),
                   stream);
  }
}  // namespace memory

GCXX_NAMESPACE_MAIN_END()


#endif