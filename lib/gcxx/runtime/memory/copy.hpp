#pragma once
#ifndef GCXX_API_RUNTIME_MEMORY_COPY_HPP_
#define GCXX_API_RUNTIME_MEMORY_COPY_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/memory/span/span.hpp>
#include <gcxx/runtime/runtime_error.hpp>
#include <gcxx/runtime/stream.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_NAMESPACE_DETAILS_BEGIN

GCXX_FH auto copy(void* desstination, const void* source,
                  const std::size_t countinBytes) -> void {
  GCXX_SAFE_RUNTIME_CALL(Memcpy, "Failed to perform async GPU copy",
                         desstination, source, countinBytes,
                         GCXX_RUNTIME_BACKEND(MemcpyDefault));
}

GCXX_FH auto copy(void* desstination, const void* source,
                  const std::size_t countinBytes,
                  const stream_wrap& stream) -> void {
  GCXX_SAFE_RUNTIME_CALL(MemcpyAsync, "Failed to perform async GPU copy",
                         desstination, source, countinBytes,
                         GCXX_RUNTIME_BACKEND(MemcpyDefault), stream.get());
}

GCXX_NAMESPACE_DETAILS_END

namespace memory {


  template <typename VT>
  GCXX_FH auto copy(VT* desstination, const VT* source,
                    const std::size_t numElements) -> void {
    details_::copy(static_cast<void*>(desstination),
                   static_cast<const void*>(source), numElements * sizeof(VT));
  }

  template <typename VT>
  GCXX_FH auto copy(const VT* desstination, const VT* source,
                    const std::size_t numElements,
                    const stream_wrap& stream) -> void {
    details_::copy(static_cast<void*>(desstination),
                   static_cast<const void*>(source), numElements * sizeof(VT),
                   stream);
  }

  template <typename VT>
  GCXX_FH auto copy(span<VT> desstination, const span<VT>& source) -> void {
    details_::copy(static_cast<void*>(desstination),
                   static_cast<const void*>(source),
                   desstination.size() * sizeof(VT));
  }

  template <typename VT>
  GCXX_FH auto copy(const span<VT>& desstination, const span<VT>& source,
                    const stream_wrap& stream) -> void {
    details_::copy(static_cast<void*>(desstination.data()),
                   static_cast<const void*>(source.data()),
                   desstination.size() * sizeof(VT), stream);
  }


}  // namespace memory

GCXX_NAMESPACE_MAIN_END


#endif