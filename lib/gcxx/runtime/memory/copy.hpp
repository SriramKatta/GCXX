#pragma once
#ifndef GCXX_API_RUNTIME_MEMORY_COPY_HPP_
#define GCXX_API_RUNTIME_MEMORY_COPY_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/memory/smartpointers/pointers.hpp>
#include <gcxx/runtime/memory/span/span.hpp>
#include <gcxx/runtime/runtime_error.hpp>
#include <gcxx/runtime/stream.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_NAMESPACE_DETAILS_BEGIN

// ╔════════════════════════════════════════════════════════╗
// ║          works on pointer with bytes to copy           ║
// ╚════════════════════════════════════════════════════════╝

GCXX_FH auto copy(void* destination, const void* source,
                  const std::size_t countinBytes) -> void {
  GCXX_SAFE_RUNTIME_CALL(Memcpy, "Failed to perform async GPU copy",
                         destination, source, countinBytes,
                         GCXX_RUNTIME_BACKEND(MemcpyDefault));
}

GCXX_FH auto copy(void* destination, const void* source,
                  const std::size_t countinBytes, const StreamView& stream)
  -> void {
  GCXX_SAFE_RUNTIME_CALL(MemcpyAsync, "Failed to perform async GPU copy",
                         destination, source, countinBytes,
                         GCXX_RUNTIME_BACKEND(MemcpyDefault), stream.getRawStream());
}

GCXX_NAMESPACE_DETAILS_END

namespace memory {

  // ╔════════════════════════════════════════════════════════╗
  // ║         pointer version based on element type          ║
  // ╚════════════════════════════════════════════════════════╝
  template <typename VT>
  GCXX_FH auto copy(VT* destination, const VT* source,
                    const std::size_t numElements) -> void {
    details_::copy((void*)destination, (const void*)source,
                   numElements * sizeof(VT));
  }

  template <typename VT>
  GCXX_FH auto copy(const VT* destination, const VT* source,
                    const std::size_t numElements, const StreamView& stream)
    -> void {
    details_::copy((void*)destination, (const void*)source,
                   numElements * sizeof(VT), stream);
  }

  // ╔════════════════════════════════════════════════════════╗
  // ║      smart pointer version based on element type       ║
  // ╚════════════════════════════════════════════════════════╝

  template <typename VT, typename DT>
  GCXX_FH auto copy(gcxx_unique_ptr<VT, DT>& destination,
                    const gcxx_unique_ptr<VT, DT>& source,
                    const std::size_t numElements) -> void {
    details_::copy(destination.get(), source.get(), numElements * sizeof(VT));
  }

  template <typename VT, typename DT>
  GCXX_FH auto copy(gcxx_unique_ptr<VT, DT>& destination,
                    const gcxx_unique_ptr<VT, DT>& source,
                    const std::size_t numElements, const StreamView& stream)
    -> void {
    details_::copy(destination.get(), source.get(), numElements * sizeof(VT),
                   stream);
  }

  // ╔════════════════════════════════════════════════════════╗
  // ║                 works on span variants                 ║
  // ╚════════════════════════════════════════════════════════╝
  template <typename VT>
  GCXX_FH auto copy(span<VT> destination, const span<VT> source) -> void {
    details_::copy(destination.data(), source.data(), destination.size_bytes());
  }

  template <typename VT>
  GCXX_FH auto copy(const span<VT> destination, const span<VT> source,
                    const StreamView& stream) -> void {
    details_::copy(destination.data(), source.data(), destination.size_bytes(),
                   stream);
  }


}  // namespace memory

GCXX_NAMESPACE_MAIN_END


#endif