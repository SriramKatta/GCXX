#pragma once
#ifndef GPUCXX_DEFINE_MACROS_HPP
#define GPUCXX_DEFINE_MACROS_HPP

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

#else  // __GPUCXXCC__

#error "Define either GPUCXX_CUDA_MODE or GPUCXX_HIP_MODE"

#endif  // __GPUCXXCC__

#endif