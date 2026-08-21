// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_MACROS_BACKEND_VERSION_MACROS_HPP_
#define GCXX_MACROS_BACKEND_VERSION_MACROS_HPP_

// CUDA version macros.
#if defined(__CUDACC__) && defined(__CUDACC_VER_MAJOR__)

#define GCXX_MAKE_CUDA_VERSION(major, minor, build) \
  (((major) * 10000000) + ((minor) * 100000) + (build))

#define GCXX_CUDA_VERSION                                            \
  GCXX_MAKE_CUDA_VERSION(__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, \
                         __CUDACC_VER_BUILD__)

#define GCXX_CUDA_VERSION_EQUALS(major, minor, build) \
  (GCXX_CUDA_VERSION == GCXX_MAKE_CUDA_VERSION(major, minor, build))

#define GCXX_CUDA_VERSION_GREATER_THAN(major, minor, build) \
  (GCXX_CUDA_VERSION > GCXX_MAKE_CUDA_VERSION(major, minor, build))

#define GCXX_CUDA_VERSION_LESS_THAN(major, minor, build) \
  (GCXX_CUDA_VERSION < GCXX_MAKE_CUDA_VERSION(major, minor, build))

#define GCXX_CUDA_VERSION_GREATER_EQUAL(major, minor, build) \
  (GCXX_CUDA_VERSION >= GCXX_MAKE_CUDA_VERSION(major, minor, build))

#define GCXX_CUDA_VERSION_LESS_EQUAL(major, minor, build) \
  (GCXX_CUDA_VERSION <= GCXX_MAKE_CUDA_VERSION(major, minor, build))

#else

#define GCXX_CUDA_VERSION 0
#define GCXX_MAKE_CUDA_VERSION(major, minor, build) 0
#define GCXX_CUDA_VERSION_EQUALS(major, minor, build) 0
#define GCXX_CUDA_VERSION_GREATER_THAN(major, minor, build) 0
#define GCXX_CUDA_VERSION_LESS_THAN(major, minor, build) 0
#define GCXX_CUDA_VERSION_GREATER_EQUAL(major, minor, build) 0
#define GCXX_CUDA_VERSION_LESS_EQUAL(major, minor, build) 0

#endif


// HIP version macros.
#if defined(__HIPCC__) && defined(HIP_VERSION)

// HIP_VERSION format: major * 10000000 + minor * 100000 + patch.
#define GCXX_HIP_VERSION HIP_VERSION

#define GCXX_MAKE_HIP_VERSION(major, minor, patch) \
  (((major) * 10000000) + ((minor) * 100000) + (patch))

#define GCXX_HIP_VERSION_EQUALS(major, minor, patch) \
  (GCXX_HIP_VERSION == GCXX_MAKE_HIP_VERSION(major, minor, patch))

#define GCXX_HIP_VERSION_GREATER_THAN(major, minor, patch) \
  (GCXX_HIP_VERSION > GCXX_MAKE_HIP_VERSION(major, minor, patch))

#define GCXX_HIP_VERSION_LESS_THAN(major, minor, patch) \
  (GCXX_HIP_VERSION < GCXX_MAKE_HIP_VERSION(major, minor, patch))

#define GCXX_HIP_VERSION_GREATER_EQUAL(major, minor, patch) \
  (GCXX_HIP_VERSION >= GCXX_MAKE_HIP_VERSION(major, minor, patch))

#define GCXX_HIP_VERSION_LESS_EQUAL(major, minor, patch) \
  (GCXX_HIP_VERSION <= GCXX_MAKE_HIP_VERSION(major, minor, patch))

#else

#define GCXX_HIP_VERSION 0
#define GCXX_MAKE_HIP_VERSION(major, minor, patch) 0
#define GCXX_HIP_VERSION_EQUALS(major, minor, patch) 0
#define GCXX_HIP_VERSION_GREATER_THAN(major, minor, patch) 0
#define GCXX_HIP_VERSION_LESS_THAN(major, minor, patch) 0
#define GCXX_HIP_VERSION_GREATER_EQUAL(major, minor, patch) 0
#define GCXX_HIP_VERSION_LESS_EQUAL(major, minor, patch) 0

#endif

#define GCXX_DEVICE_COMPILE (__HIP_DEVICE_COMPILE__ || __CUDA_ARCH__)

#endif
