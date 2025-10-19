#pragma once
#ifndef GCXX_TYPES_SHARED_MEM_HPP
#define GCXX_TYPES_SHARED_MEM_HPP


#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

template <typename VT>
struct SharedMemory {
  __device__ inline operator VT*() {
    extern __shared__ std::byte __smem[];
    return (VT*)__smem;
  }

  __device__ inline operator const VT*() const {
    extern __shared__ std::byte __smem[];
    return (VT*)__smem;
  }
};

template <>
struct SharedMemory<double> {
  __device__ inline operator double*() {
    extern __shared__ std::byte __smem[];
    return (double*)__smem;
  }

  __device__ inline operator const double*() const {
    extern __shared__ std::byte __smem[];
    return (double*)__smem;
  }
};

GCXX_NAMESPACE_MAIN_END


#endif