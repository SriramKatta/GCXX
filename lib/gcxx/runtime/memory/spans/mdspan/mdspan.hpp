// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MDSPAN_MDSPAN_HPP
#define GCXX_RUNTIME_MEMORY_MDSPAN_MDSPAN_HPP

#define MDSPAN_IMPL_STANDARD_NAMESPACE gcxx
#if defined(CMAKE_GCXX_CUDA_MODE)
#define MDSPAN_IMPL_HAS_CUDA 1
#endif
#if defined(CMAKE_GCXX_HIP_MODE)
#define MDSPAN_IMPL_HAS_HIP 1
#endif

#include <mdspan.hpp>

#undef MDSPAN_IMPL_STANDARD_NAMESPACE
#if defined(CMAKE_GCXX_CUDA_MODE)
#undef MDSPAN_IMPL_HAS_CUDA
#endif
#if defined(CMAKE_GCXX_HIP_MODE)
#undef MDSPAN_IMPL_HAS_HIP
#endif


#include <gcxx/runtime/memory/spans/mdspan/resrict_accessor.hpp>

#endif