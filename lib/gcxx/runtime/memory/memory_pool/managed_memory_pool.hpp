// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Derived from NVIDIA CCCL libcudacxx:
//   https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__memory_resource/__memory_pool/managed_memory_pool.h
// Copyright (c) NVIDIA Corporation and affiliates.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// managed_memory_pool(_ref): a managed (unified) memory pool. Allocations are
// accessible from both host and device. CUDA 13.0+ only (managed memory pools
// are not supported before CUDA 13.0 / not on Windows). The owning pool creates
// a cudaMemPool_t with location type None and allocation type Managed.
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMORY_POOL_MANAGED_MEMORY_POOL_HPP_
#define GCXX_RUNTIME_MEMORY_MEMORY_POOL_MANAGED_MEMORY_POOL_HPP_

#include <cstddef>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/memory/buffers/no_init.hpp>
#include <gcxx/runtime/memory/buffers/properties.hpp>
#include <gcxx/runtime/memory/memory_pool/memory_pool_base.hpp>
#include <gcxx/runtime/memory/memory_pool/memory_pool_properties.hpp>
#include <gcxx/runtime/memory/memory_resource/resource_concepts.hpp>
#include <gcxx/runtime_backend/backend_stream_memory.hpp>

#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()

// managed_memory_pool_ref: non-owning managed memory pool. Managed memory is
// reachable from both host and device, so it advertises both properties.
class managed_memory_pool_ref : public memory_pool_base {
 public:
  using properties = TypeSet<device_accessible, host_accessible>;

  GCXX_FH explicit managed_memory_pool_ref(driver::deviceMemPool_t pool) noexcept
      : memory_pool_base(pool) {}

  using memory_pool_base::allocate;
  using memory_pool_base::deallocate;

  // gcxx resource-concept adapters (bytes-first arg order).
  GCXX_FH auto allocate(std::size_t num_bytes, gcxx::StreamView stream) -> void* {
    return memory_pool_base::allocate(stream, num_bytes);
  }
  GCXX_FH auto deallocate(void* ptr, gcxx::StreamView stream) -> void {
    memory_pool_base::deallocate(stream, ptr, 0);
  }
};

// managed_memory_pool: owning managed memory pool. Managed memory has no fixed
// device placement (location type None), so the ctor takes no device id.
struct managed_memory_pool : managed_memory_pool_ref {
  using reference_type = managed_memory_pool_ref;

  GCXX_FH explicit managed_memory_pool(no_init_t) noexcept
      : managed_memory_pool_ref(driver::deviceMemPool_t{}) {}

  GCXX_FH managed_memory_pool(memory_pool_properties props = {})
      : managed_memory_pool_ref(create_memory_pool(flags::MemLocation::None, 0,
                                                   flags::MemAllocation::Managed,
                                                   props)) {}

  GCXX_FH ~managed_memory_pool() noexcept {
    if (m_pool_ != nullptr) {
      driver::deviceMemPoolDestroy(m_pool_);
    }
  }

  GCXX_FH static auto from_native_handle(driver::deviceMemPool_t pool) noexcept
    -> managed_memory_pool {
    return managed_memory_pool(pool);
  }

  GCXX_FH constexpr auto release() noexcept -> driver::deviceMemPool_t {
    driver::deviceMemPool_t pool = m_pool_;
    m_pool_                      = nullptr;
    return pool;
  }

  GCXX_FH auto as_ref() noexcept -> managed_memory_pool_ref& {
    return static_cast<managed_memory_pool_ref&>(*this);
  }

  managed_memory_pool(const managed_memory_pool&)            = delete;
  managed_memory_pool& operator=(const managed_memory_pool&) = delete;

 private:
  GCXX_FH explicit managed_memory_pool(driver::deviceMemPool_t pool) noexcept
      : managed_memory_pool_ref(pool) {}
};

static_assert(resource_with<managed_memory_pool_ref, device_accessible, host_accessible>,
              "managed_memory_pool_ref must model the gcxx resource concept");
static_assert(resource_with<managed_memory_pool, device_accessible, host_accessible>,
              "managed_memory_pool must model the gcxx resource concept");

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif  // GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)

#endif
