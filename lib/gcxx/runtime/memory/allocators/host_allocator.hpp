#pragma once
#ifndef GCXX_RUNTIME_MEMORY_ALLOCATOR_HOST_ALLOCATOR_HPP
#define GCXX_RUNTIME_MEMORY_ALLOCATOR_HOST_ALLOCATOR_HPP


#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

template <class VT>
class host_allocator {

  public:
  using value_type = VT;

  host_allocator() noexcept = default;

  template <class U>
  constexpr host_allocator(const host_allocator<U>&) noexcept {}

  [[nodiscard]]
  VT* allocate(std::size_t n) {
    void* ptr;
    GCXX_SAFE_RUNTIME_CALL(
#if GCXX_CUDA_MODE
      MallocHost
#elif GCXX_HIP_MODE
      HostMalloc
#endif
      ,
      "Failed to allocate Pinned host memory", &ptr, n * sizeof(VT));
    return static_cast<VT*>(ptr);
  }

  void deallocate(VT* p, std::size_t) noexcept {
    GCXX_SAFE_RUNTIME_CALL(FreeHost, "Failed to deallocate Pinned host memory",
                           p);
  }

  // Stateless allocators compare equal
  template <class U>
  constexpr bool operator==(const host_allocator<U>&) const noexcept {
    return true;
  }

  template <class U>
  constexpr bool operator!=(const host_allocator<U>&) const noexcept {
    return false;
  }
};

GCXX_NAMESPACE_MAIN_DETAILS_END


#endif