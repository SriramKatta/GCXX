#pragma once
#ifndef GCXX_API_RUNTIME_MEMORY_COPY_HPP_
#define GCXX_API_RUNTIME_MEMORY_COPY_HPP_

#include <gcxx/internal/prologue.hpp>
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
  // ║      smart pointer version based on element type       ║
  // ╚════════════════════════════════════════════════════════╝

  template <typename VT, typename DTDest, typename DTSource>
  GCXX_FH auto Copy(gcxx_unique_ptr<VT, DTDest>& destination,
                    const gcxx_unique_ptr<VT, DTSource>& source,
                    const std::size_t numElements) -> void {
    details_::Copy(destination.get(), source.get(), numElements * sizeof(VT));
  }

  template <typename VT, typename DTDest, typename DTSource>
  GCXX_FH auto Copy(gcxx_unique_ptr<VT, DTDest>& destination,
                    const gcxx_unique_ptr<VT, DTSource>& source,
                    const std::size_t numElements, const StreamView& stream)
    -> void {
    details_::Copy(destination.get(), source.get(), numElements * sizeof(VT),
                   stream);
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


}  // namespace memory

GCXX_NAMESPACE_MAIN_END


#endif