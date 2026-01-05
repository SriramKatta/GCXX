#pragma once
#ifndef GCXX_API_RUNTIME_MEMORY_MEMSET_HPP_
#define GCXX_API_RUNTIME_MEMORY_MEMSET_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/memory/smartpointers/pointers.hpp>
#include <gcxx/runtime/memory/spans/spans.hpp>
#include <gcxx/runtime/runtime_error.hpp>
#include <gcxx/runtime/stream.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_NAMESPACE_DETAILS_BEGIN

// ╔════════════════════════════════════════════════════════╗
// ║          works on pointer with bytes to memset           ║
// ╚════════════════════════════════════════════════════════╝

GCXX_FH auto Memset(void* dev_ptr, const int value,
                    const std::size_t countinBytes) -> void {
  GCXX_SAFE_RUNTIME_CALL(Memset, "Failed to perform GPU memset", dev_ptr, value,
                         countinBytes);
}

GCXX_FH auto Memset(void* dev_ptr, const int value,
                    const std::size_t countinBytes, const StreamView& stream)
  -> void {
  GCXX_SAFE_RUNTIME_CALL(MemsetAsync, "Failed to perform Async GPU memset",
                         dev_ptr, value, countinBytes, stream.getRawStream());
}

GCXX_NAMESPACE_DETAILS_END

namespace memory {

  // ╔════════════════════════════════════════════════════════╗
  // ║         pointer version based on element type          ║
  // ╚════════════════════════════════════════════════════════╝
  template <typename VT>
  GCXX_FH auto Memset(VT* dev_ptr, const int value,
                      const std::size_t numElements) -> void {
    details_::Memset((void*)dev_ptr, value, numElements * sizeof(VT));
  }

  template <typename VT>
  GCXX_FH auto Memset(const VT* dev_ptr, const int value,
                      const std::size_t numElements, const StreamView& stream)
    -> void {
    details_::Memset((void*)dev_ptr, value, numElements * sizeof(VT), stream);
  }

  // ╔════════════════════════════════════════════════════════╗
  // ║      smart pointer version based on element type       ║
  // ╚════════════════════════════════════════════════════════╝

  template <typename VT>
  GCXX_FH auto Memset(device_ptr<VT>& destination, const int value,
                      const std::size_t numElements) -> void {
    details_::Memset(destination.get(), value, numElements * sizeof(VT));
  }

  template <typename VT>
  GCXX_FH auto Memset(device_ptr<VT>& destination, const int value,
                      const std::size_t numElements, const StreamView& stream)
    -> void {
    details_::Memset(destination.get(), value, numElements * sizeof(VT),
                     stream);
  }

  // ╔════════════════════════════════════════════════════════╗
  // ║                 works on span variants                 ║
  // ╚════════════════════════════════════════════════════════╝
  template <typename VT>
  GCXX_FH auto Memset(span<VT> destination, const int value) -> void {
    details_::Memset(destination.data(), value, destination.size_bytes());
  }

  template <typename VT>
  GCXX_FH auto Memset(const span<VT> destination, const int value,
                      const StreamView& stream) -> void {
    details_::Memset(destination.data(), value, destination.size_bytes(),
                     stream);
  }


}  // namespace memory

GCXX_NAMESPACE_MAIN_END


#endif