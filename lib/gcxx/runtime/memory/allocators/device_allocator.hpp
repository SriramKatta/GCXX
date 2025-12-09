#pragma once
#ifndef GCXX_RUNTIME_MEMORY_ALLOCATOR_DEVICE_ALLOCATOR_HPP
#define GCXX_RUNTIME_MEMORY_ALLOCATOR_DEVICE_ALLOCATOR_HPP


#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/details/device_memory_helper.hpp>


GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

template <class VT>
class device_allocator {
 public:
  using value_type = VT;

  device_allocator() noexcept = default;

  template <class U>
  constexpr device_allocator(const device_allocator<U>&) noexcept {}

  [[nodiscard]] VT* allocate(std::size_t n) {
    return static_cast<VT*>(details_::device_malloc(n * sizeof(VT)));
  }

  void deallocate(VT* p, std::size_t) noexcept { details_::device_free(p); }

  // Stateless allocators compare equal
  template <class U>
  constexpr bool operator==(const device_allocator<U>&) const noexcept {
    return true;
  }

  template <class U>
  constexpr bool operator!=(const device_allocator<U>&) const noexcept {
    return false;
  }

  // needed since std::vector by default constructs the object on host
  // irrespective of memory space
  template <typename U>
  constexpr void construct(U* p) {
    // ::new (static_cast<void*>(p)) U;
    // GCXX_SAFE_RUNTIME_CALL(Memset, "Failed to memset the data", p, 0,
    // sizeof(U));
  }
};

GCXX_NAMESPACE_MAIN_DETAILS_END


#endif