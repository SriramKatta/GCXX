// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_MACROS_BACKEND_MODE_MACROS_HPP_
#define GCXX_MACROS_BACKEND_MODE_MACROS_HPP_

#if defined(CMAKE_GCXX_CUDA_MODE)
#define GCXX_CUDA_MODE() 1
#else
#define GCXX_CUDA_MODE() 0
#endif

#if defined(CMAKE_GCXX_HIP_MODE)
#define GCXX_HIP_MODE() 1
#else
#define GCXX_HIP_MODE() 0
#endif

#endif
