#pragma once
#ifndef GCXX_API_RUNTIME_MEMORY_COPY_HPP_
#define GCXX_API_RUNTIME_MEMORY_COPY_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/macros/template_helper_macros.hpp>
#include <gcxx/runtime/details/helper_function.hpp>
#include <gcxx/runtime/memory/smartpointers/pointers.hpp>
#include <gcxx/runtime/memory/spans/spans.hpp>
#include <gcxx/runtime/stream.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_NAMESPACE_DETAILS_BEGIN

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
                  const std::size_t countinBytes, const StreamView& stream)
  -> void {
  GCXX_SAFE_RUNTIME_CALL(
    MemcpyAsync, "Failed to perform async GPU copy", destination, source,
    countinBytes, GCXX_RUNTIME_BACKEND(MemcpyDefault), stream.getRawStream());
}

GCXX_NAMESPACE_DETAILS_END

namespace memory {

  // ╔════════════════════════════════════════════════════════╗
  // ║         pointer version based on element type          ║
  // ╚════════════════════════════════════════════════════════╝
  template <typename VT>
  GCXX_FH auto Copy(VT* destination, const VT* source,
                    const std::size_t numElements) -> void {
    details_::Copy((void*)destination, (const void*)source,
                   numElements * sizeof(VT));
  }

  template <typename VT>
  GCXX_FH auto Copy(const VT* destination, const VT* source,
                    const std::size_t numElements, const StreamView& stream)
    -> void {
    details_::Copy((void*)destination, (const void*)source,
                   numElements * sizeof(VT), stream);
  }

  // ╔════════════════════════════════════════════════════════╗
  // ║                 works on span variants                 ║
  // ╚════════════════════════════════════════════════════════╝
  template <typename VT>
  GCXX_FH auto Copy(span<VT> destination, const span<VT> source) -> void {
    details_::Copy(destination.data(), source.data(), destination.size_bytes());
  }

  template <typename VT>
  GCXX_FH auto Copy(const span<VT> destination, const span<VT> source,
                    const StreamView& stream) -> void {
    details_::Copy(destination.data(), source.data(), destination.size_bytes(),
                   stream);
  }

  // ╔════════════════════════════════════════════════════════╗
  // ║    works on any type with data() and size() methods    ║
  // ╚════════════════════════════════════════════════════════╝
  GCXX_TEMPLATE(typename DSTTY, typename SRCTY)
  GCXX_REQUIRES(
    details_::has_size_and_data_v<details_::uncvref_t<DSTTY>> GCXX_AND
      details_::has_size_and_data_v<details_::uncvref_t<SRCTY>>)

  GCXX_FH auto Copy(DSTTY&& destination, SRCTY&& source) -> void {
    using dst_element_type =
      details_::uncvref_t<decltype(*details_::data(destination))>;
    details_::Copy(details_::to_address(details_::data(destination)),
                   details_::to_address(details_::data(source)),
                   details_::size(destination) * sizeof(dst_element_type));
  }

  GCXX_TEMPLATE(typename DSTTY, typename SRCTY)
  GCXX_REQUIRES(
    details_::has_size_and_data_v<details_::uncvref_t<DSTTY>> GCXX_AND
      details_::has_size_and_data_v<details_::uncvref_t<SRCTY>>)

  GCXX_FH auto Copy(DSTTY&& destination, SRCTY&& source,
                    const StreamView& stream) -> void {
    using dst_element_type =
      details_::uncvref_t<decltype(*details_::data(destination))>;
    details_::Copy(details_::to_address(details_::data(destination)),
                   details_::to_address(details_::data(source)),
                   details_::size(destination) * sizeof(dst_element_type),
                   stream);
  }

}  // namespace memory

GCXX_NAMESPACE_MAIN_END


#endif