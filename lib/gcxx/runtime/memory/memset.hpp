// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_API_RUNTIME_MEMORY_MEMSET_HPP_
#define GCXX_API_RUNTIME_MEMORY_MEMSET_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/memory/smartpointers/pointers.hpp>
#include <gcxx/runtime/memory/spans/spans.hpp>
#include <gcxx/runtime/stream.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>
#include <gcxx/runtime_backend/backend_memory.hpp>
#include <type_traits>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_DETAILS_BEGIN()

// ╔════════════════════════════════════════════════════════╗
// ║          works on pointer with bytes to memset           ║
// ╚════════════════════════════════════════════════════════╝

GCXX_FH auto Memset(void* dev_ptr, const int value,
                    const std::size_t countinBytes) -> void {
  driver::deviceMemset(dev_ptr, value, countinBytes);
}

GCXX_FH auto Memset(void* dev_ptr, const int value,
                    const std::size_t countinBytes,
                    const StreamView& stream) -> void {
  driver::deviceMemsetAsync(dev_ptr, value, countinBytes,
                            stream.getRawStream());
}

GCXX_NAMESPACE_DETAILS_END()

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
                      const std::size_t numElements,
                      const StreamView& stream) -> void {
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
                      const std::size_t numElements,
                      const StreamView& stream) -> void {
    details_::Memset(destination.get(), value, numElements * sizeof(VT),
                     stream);
  }

  // Generic pointer-like handle (e.g., std::unique_ptr with custom deleter)
  template <typename Ptr,
            typename = std::void_t<decltype(std::declval<Ptr&>().get())>>
  GCXX_FH auto Memset(Ptr& handle, const int value,
                      const std::size_t numElements) -> void {
    using raw_ptr_t = decltype(std::declval<Ptr&>().get());
    using VT        = std::remove_pointer_t<std::remove_cv_t<raw_ptr_t>>;
    details_::Memset(handle.get(), value, numElements * sizeof(VT));
  }

  template <typename Ptr,
            typename = std::void_t<decltype(std::declval<Ptr&>().get())>>
  GCXX_FH auto Memset(Ptr& handle, const int value,
                      const std::size_t numElements,
                      const StreamView& stream) -> void {
    using raw_ptr_t = decltype(std::declval<Ptr&>().get());
    using VT        = std::remove_pointer_t<std::remove_cv_t<raw_ptr_t>>;
    details_::Memset(handle.get(), value, numElements * sizeof(VT), stream);
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

GCXX_NAMESPACE_MAIN_END()


#endif