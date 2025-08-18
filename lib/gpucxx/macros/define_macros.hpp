#pragma once
#ifndef GPUCXX_MACROS_DEFINE_MACROS_HPP_
#define GPUCXX_MACROS_DEFINE_MACROS_HPP_

#if defined(GPUCXX_CUDA_MODE) || defined(GPUCXX_HIP_MODE)

#ifndef GPUCXX_H
#define GPUCXX_H __host__
#endif

#ifndef GPUCXX_FH
#define GPUCXX_FH __forceinline__ __host__
#endif

#ifndef GPUCXX_D
#define GPUCXX_D __device__
#endif

#ifndef GPUCXX_FD
#define GPUCXX_FD __forceinline__ __device__
#endif

#ifndef GPUCXX_HD
#define GPUCXX_HD __host__ __device__
#endif

#ifndef GPUCXX_FHD
#define GPUCXX_FHD __forceinline__ __host__ __device__
#endif

#ifndef GPUCXX_CA
#define GPUCXX_CA constexpr auto
#endif

#ifndef GPUCXX_NOEXCEPT
#define GPUCXX_NOEXCEPT noexcept
#endif


#ifndef GPUCXX_CONST_NOEXCEPT
#define GPUCXX_CONST_NOEXCEPT const noexcept
#endif

using flag_t = unsigned int;


#ifndef GPUCXX_BEGIN_NAMESPACE
#define GPUCXX_BEGIN_NAMESPACE \
  namespace gcxx {             \
    inline namespace v1 {
#define GPUCXX_END_NAMESPACE   \
  } /* inline namespace v1  */ \
  }  // namespace gcxx
#endif

#ifndef GPUCXX_DETAILS_BEGIN_NAMESPACE
#define GPUCXX_DETAILS_BEGIN_NAMESPACE \
  GPUCXX_BEGIN_NAMESPACE               \
  namespace details_ {
#define GPUCXX_DETAILS_END_NAMESPACE \
  } /* namespace details_  */        \
  GPUCXX_END_NAMESPACE               
#endif

#endif  // defined(GPUCXX_CUDA_MODE) || defined(GPUCXX_HIP_MODE)
#endif
