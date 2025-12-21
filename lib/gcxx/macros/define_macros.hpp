#pragma once
#ifndef GCXX_MACROS_DEFINE_MACROS_HPP_
#define GCXX_MACROS_DEFINE_MACROS_HPP_

#if defined(GCXX_CUDA_MODE) || defined(GCXX_HIP_MODE)

#ifndef GCXX_H
#define GCXX_H __host__
#endif

#ifndef GCXX_FH
#define GCXX_FH __forceinline__ __host__
#endif

#ifndef GCXX_FC
#define GCXX_FC __forceinline__ constexpr
#endif

#ifndef GCXX_FHC
#define GCXX_FHC __forceinline__ __host__ constexpr
#endif

#ifndef GCXX_D
#define GCXX_D __device__
#endif

#ifndef GCXX_FD
#define GCXX_FD __forceinline__ __device__
#endif

#ifndef GCXX_FDC
#define GCXX_FDC __forceinline__ __device__ constexpr
#endif

#ifndef GCXX_HD
#define GCXX_HD __host__ __device__
#endif

#ifndef GCXX_FHD
#define GCXX_FHD __forceinline__ __host__ __device__
#endif

#ifndef GCXX_FHDC
#define GCXX_FHDC __forceinline__ __host__ __device__ constexpr
#endif

#ifndef GCXX_CXPR
#define GCXX_CXPR constexpr
#endif

#ifndef GCXX_NOEXCEPT
#define GCXX_NOEXCEPT noexcept
#endif


#ifndef GCXX_CONST_NOEXCEPT
#define GCXX_CONST_NOEXCEPT const noexcept
#endif

#if GCXX_CUDA_MODE
#define GCXXRT_CB CUDART_CB
#else
#define GCXXRT_CB
#endif

#ifndef GCXX_NAMESPACE_MAIN_BEGIN
#define GCXX_NAMESPACE_MAIN_BEGIN \
  namespace gcxx {                \
    inline namespace v1 {
#define GCXX_NAMESPACE_MAIN_END \
  }  /* inline namespace v1  */ \
  }  // namespace gcxx
#endif

#ifndef GCXX_NAMESPACE_DETAILS_BEGIN
#define GCXX_NAMESPACE_DETAILS_BEGIN namespace details_ {
#define GCXX_NAMESPACE_DETAILS_END } /* namespace details_  */
#endif

#ifndef GCXX_NAMESPACE_FLAGS_BEGIN
#define GCXX_NAMESPACE_FLAGS_BEGIN namespace flags {
#define GCXX_NAMESPACE_FLAGS_END } /* namespace flags  */
#endif


#ifndef GCXX_NAMESPACE_MAIN_DETAILS_BEGIN
#define GCXX_NAMESPACE_MAIN_DETAILS_BEGIN \
  GCXX_NAMESPACE_MAIN_BEGIN               \
  GCXX_NAMESPACE_DETAILS_BEGIN
#define GCXX_NAMESPACE_MAIN_DETAILS_END \
  GCXX_NAMESPACE_DETAILS_END            \
  GCXX_NAMESPACE_MAIN_END
#endif

#ifndef GCXX_NAMESPACE_MAIN_FLAGS_BEGIN
#define GCXX_NAMESPACE_MAIN_FLAGS_BEGIN \
  GCXX_NAMESPACE_MAIN_BEGIN             \
  GCXX_NAMESPACE_FLAGS_BEGIN
#define GCXX_NAMESPACE_MAIN_FLAGS_END \
  GCXX_NAMESPACE_FLAGS_END            \
  GCXX_NAMESPACE_MAIN_END
#endif

GCXX_NAMESPACE_MAIN_BEGIN
using device_t = int;
GCXX_NAMESPACE_MAIN_END

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN
using flag_t = unsigned int;
GCXX_NAMESPACE_MAIN_DETAILS_END

#if defined(__INTEL_COMPILER)
#define GCXX_RESTRICT_KEYWORD __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define GCXX_RESTRICT_KEYWORD __restrict__
#else
#define GCXX_RESTRICT_KEYWORD
#endif

#endif  // defined(GCXX_CUDA_MODE) || defined(GCXX_HIP_MODE)
#endif
