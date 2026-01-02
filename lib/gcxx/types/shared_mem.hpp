#pragma once
#ifndef GCXX_TYPES_SHARED_MEM_HPP
#define GCXX_TYPES_SHARED_MEM_HPP

#include <cstddef>

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>


GCXX_NAMESPACE_MAIN_BEGIN

// NOLINTBEGIN
template <typename VT>
struct dynamicSharedMemory {
  __device__ inline operator VT*() {
    extern __shared__ char m_smem[];
    return (VT*)m_smem;
  }

  __device__ inline operator const VT*() const {
    extern __shared__ char m_smem[];
    return (VT*)m_smem;
  }
};

template <>
struct dynamicSharedMemory<double> {
  __device__ inline operator double*() {
    extern __shared__ char m_smem[];
    return (double*)m_smem;
  }

  __device__ inline operator const double*() const {
    extern __shared__ char m_smem[];
    return (double*)m_smem;
  }
};

template <typename VT, std::size_t N>
struct staticSharedArray {
  __device__ inline operator VT*() {
    extern __shared__ VT m_smem[];
    return (VT*)m_smem;
  }

  __device__ inline operator const VT*() const {
    extern __shared__ VT m_smem[];
    return (VT*)m_smem;
  }
};

// NOLINTEND
GCXX_NAMESPACE_MAIN_END


#endif