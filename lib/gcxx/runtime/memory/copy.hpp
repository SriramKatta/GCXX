#pragma once
#ifndef GCXX_API_RUNTIME_MEMORY_COPY_HPP_
#define GCXX_API_RUNTIME_MEMORY_COPY_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/runtime_error.hpp>
#include <gcxx/runtime/span/span.hpp>
#include <gcxx/runtime/stream.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_NAMESPACE_DETAILS_BEGIN

GCXX_FH auto copy(void* dst, const void* src, const std::size_t countinBytes)
  -> void {
  GCXX_SAFE_RUNTIME_CALL(Memcpy, "Failed to perform async GPU copy", dst, src,
                         countinBytes, GCXX_RUNTIME_BACKEND(MemcpyDefault));
}

GCXX_FH auto copy(void* dst, const void* src, const std::size_t countinBytes,
                  const stream_wrap& stream) -> void {
  GCXX_SAFE_RUNTIME_CALL(MemcpyAsync, "Failed to perform async GPU copy", dst,
                         src, countinBytes, GCXX_RUNTIME_BACKEND(MemcpyDefault),
                         stream.get());
}

GCXX_NAMESPACE_DETAILS_END

namespace memory {


  template <typename VT>
  GCXX_FH auto copy(VT* dst, const VT* src, const std::size_t numElements)
    -> void {
    details_::copy(static_cast<void*>(dst), static_cast<const void*>(src),
                   numElements * sizeof(VT));
  }

  template <typename VT>
  GCXX_FH auto copy(const VT* dst, const VT* src, const std::size_t numElements,
                    const stream_wrap& stream) -> void {
    details_::copy(static_cast<void*>(dst), static_cast<const void*>(src),
                   numElements * sizeof(VT), stream);
  }

  template <typename VT>
  GCXX_FH auto copy(span<VT> dst, const span<VT>& src) -> void {
    details_::copy(static_cast<void*>(dst), static_cast<const void*>(src),
                   dst.size() * sizeof(VT));
  }

  template <typename VT>
  GCXX_FH auto copy(const span<VT>& dst, const span<VT>& src,
                    const stream_wrap& stream) -> void {
    details_::copy(static_cast<void*>(dst.data()),
                   static_cast<const void*>(src.data()),
                   dst.size() * sizeof(VT), stream);
  }


}  // namespace memory

GCXX_NAMESPACE_MAIN_END


#endif