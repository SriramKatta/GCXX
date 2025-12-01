#pragma once
#ifndef GCXX_RUNTIME_MEMORY_ALLOCATOR_DEVICE_ALLOCATOR_HPP
#define GCXX_RUNTIME_MEMORY_ALLOCATOR_DEVICE_ALLOCATOR_HPP


#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

#include <memory_resource>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

template <class VT>
class device_allocator {
 public:
  using value_type = VT;

  device_allocator() noexcept = default;

  template <class U>
  constexpr device_allocator(const device_allocator<U>&) noexcept {}

  [[nodiscard]]
  VT* allocate(std::size_t n) {
    void* ptr;
    GCXX_SAFE_RUNTIME_CALL(Malloc, "Failed to allocate device memory", &ptr,
                           n * sizeof(VT));
    return static_cast<VT*>(ptr);
  }

  void deallocate(VT* p, std::size_t) noexcept {
    GCXX_SAFE_RUNTIME_CALL(Free, "Failed to deallocate device memory", p);
  }

  // Stateless allocators compare equal
  template <class U>
  constexpr bool operator==(const device_allocator<U>&) const noexcept {
    return true;
  }

  template <class U>
  constexpr bool operator!=(const device_allocator<U>&) const noexcept {
    return false;
  }

  // needed since std::vector by default constructs the object irresoective of
  // memory space
  template <typename U>
  void construct(U* p) {
    ::new (static_cast<void*>(p)) U;
  }
};

GCXX_NAMESPACE_MAIN_DETAILS_END


#endif