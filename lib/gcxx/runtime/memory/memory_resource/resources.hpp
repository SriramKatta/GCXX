// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMORY_RESOURCE_RESOURCES_HPP_
#define GCXX_RUNTIME_MEMORY_MEMORY_RESOURCE_RESOURCES_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/memory/device_memory_helper.hpp>
#include <gcxx/runtime/memory/memory_resource/basic_resource.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// Resource aliases — instantiate basic_resource with the function objects
// from device_memory_helper.hpp. Each alias is the same shape as the old
// hand-written forwarder class it replaces.
//
// sync_*      : sync alloc + sync free. basic_resource syncs the stream
//               before each free.
// async_*     : stream-ordered alloc + free; no extra sync.
// managed_*   : bonus — was missing before T2. device_managed_malloc already
//               existed in device_memory_helper.hpp but no resource wrapped it.
// ─────────────────────────────────────────────────────────────────────────────

// cudaMalloc / cudaFree — sync.
// Replaces the old sync_device_resource class.
using sync_device_resource =
  basic_resource<details_::device_malloc_t, details_::device_free_t>;

// cudaMallocHost / cudaFreeHost — sync, host-visible.
// Replaces the old sync_host_resource class.
using sync_host_resource =
  basic_resource<details_::host_malloc_t, details_::host_free_t>;

// cudaMallocAsync / cudaFreeAsync — stream-ordered.
// Replaces the old async_device_resource class.
using async_device_resource =
  basic_resource<details_::device_malloc_async_t, details_::device_free_async_t>;

// cudaMallocManaged / cudaFree — sync, unified memory. Bonus: previously
// missing (the function object existed but no resource wrapped it).
using managed_device_resource =
  basic_resource<details_::device_managed_malloc_t, details_::device_free_t>;

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
