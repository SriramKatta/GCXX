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

using flag_t = unsigned int;


#ifndef GCXX_NAMESPACE_MAIN_BEGIN
#define GCXX_NAMESPACE_MAIN_BEGIN \
  namespace gcxx {                \
    inline namespace v1 {
#define GCXX_NAMESPACE_MAIN_END \
  } /* inline namespace v1  */  \
  }  // namespace gcxx
#endif

#ifndef GCXX_NAMESPACE_DETAILS_BEGIN
#define GCXX_NAMESPACE_DETAILS_BEGIN namespace details_ {
#define GCXX_NAMESPACE_DETAILS_END } /* namespace details_  */
#endif


#ifndef GCXX_NAMESPACE_MAIN_DETAILS_BEGIN
#define GCXX_NAMESPACE_MAIN_DETAILS_BEGIN \
  GCXX_NAMESPACE_MAIN_BEGIN               \
  GCXX_NAMESPACE_DETAILS_BEGIN
#define GCXX_NAMESPACE_MAIN_DETAILS_END \
  GCXX_NAMESPACE_DETAILS_END            \
  GCXX_NAMESPACE_MAIN_END
#endif

#endif  // defined(GCXX_CUDA_MODE) || defined(GCXX_HIP_MODE)
#endif
