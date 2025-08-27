#pragma once
#ifndef GPUCXX_API_RUNTIME_MEMORY_COPY_HPP_
#define GPUCXX_API_RUNTIME_MEMORY_COPY_HPP_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>
#include <gpucxx/runtime/runtime_error.hpp>
#include <gpucxx/runtime/stream.hpp>


GPUCXX_BEGIN_NAMESPACE

namespace memory {

  namespace details_ {

    GPUCXX_FH auto copy(void* dst, const void* src,
                        const std::size_t countinBytes) -> void {
      GPUCXX_SAFE_RUNTIME_CALL(Memcpy, (dst, src, countinBytes,
                                        GPUCXX_RUNTIME_BACKEND(MemcpyDefault)));
    }

    GPUCXX_FH auto copy(void* dst, const void* src,
                        const std::size_t countinBytes,
                        const stream_ref& stream) -> void {
      GPUCXX_SAFE_RUNTIME_CALL(
        MemcpyAsync, (dst, src, countinBytes,
                      GPUCXX_RUNTIME_BACKEND(MemcpyDefault), stream.get()));
    }
  }  // namespace details_

  template <typename VT>
  GPUCXX_FH auto copy(VT* dst, const VT* src, const std::size_t numEntries)
    -> void {
    details_::copy(static_cast<void*>(dst), static_cast<const void*>(src),
                   numEntries * sizeof(VT));
  }

  template <typename VT>
  GPUCXX_FH auto copy(VT* dst, const VT* src, const std::size_t numEntries,
                      const stream_ref& stream) -> void {
    details_::copy(static_cast<void*>(dst), static_cast<const void*>(src),
                   numEntries * sizeof(VT), stream);
  }

}  // namespace memory

GPUCXX_END_NAMESPACE


#endif