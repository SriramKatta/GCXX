// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MDSPAN_MDSPAN_HPP
#define GCXX_RUNTIME_MEMORY_MDSPAN_MDSPAN_HPP

#define MDSPAN_IMPL_STANDARD_NAMESPACE gcxx
#define MDSPAN_IMPL_HAS_CUDA 1
#define MDSPAN_IMPL_HAS_HIP 1

#include <mdspan.hpp>

#undef MDSPAN_IMPL_STANDARD_NAMESPACE
#undef MDSPAN_IMPL_HAS_CUDA
#undef MDSPAN_IMPL_HAS_HIP


#include <gcxx/runtime/memory/spans/mdspan/host_device_accessor.hpp>
#include <gcxx/runtime/memory/spans/mdspan/host_device_mdspan.hpp>
#include <gcxx/runtime/memory/spans/mdspan/resrict_accessor.hpp>
#include <gcxx/runtime/memory/spans/mdspan/shared_memory_accessor.hpp>

#endif