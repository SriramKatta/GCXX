// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMORY_RESOURCE_RESOURCES_HPP_
#define GCXX_RUNTIME_MEMORY_MEMORY_RESOURCE_RESOURCES_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/memory/device_memory_helper.hpp>
#include <gcxx/runtime/memory/buffers/properties.hpp>
#include <gcxx/runtime/memory/memory_resource/synchronous_resource.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// Resource aliases — instantiate synchronous_resource with the sync allocator
// function objects from device_memory_helper.hpp. Each alias carries its
// accessibility property (device_accessible / host_accessible) via the
// synchronous_resource Properties... pack, exposed as `using properties`.
//
// sync_*      : sync alloc + sync free. synchronous_resource syncs the stream
//               before each free (NCCL/NVSHMEM-safe; correct-but-slower for
//               plain cudaMalloc/cudaFreeHost).
// managed_*   : unified memory (both host- and device-accessible).
//
// Async/stream-ordered device allocation uses pooled_device_resource (the pool
// resource), not a wrapper here.
// ─────────────────────────────────────────────────────────────────────────────

// cudaMalloc / cudaFree — sync, device-visible.
using sync_device_resource =
  synchronous_resource<details_::device_malloc_t, details_::device_free_t,
                       device_accessible>;

// cudaMallocHost / cudaFreeHost — sync, host-visible.
using sync_host_resource =
  synchronous_resource<details_::host_malloc_t, details_::host_free_t,
                       host_accessible>;

// cudaMallocManaged / cudaFree — sync, unified memory (host- and
// device-visible).
using managed_device_resource =
  synchronous_resource<details_::device_managed_malloc_t,
                       details_::device_free_t, device_accessible,
                       host_accessible>;

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
