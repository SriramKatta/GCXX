// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Derived from NVIDIA CCCL libcudacxx:
//   https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__memory_resource/__memory_pool/managed_memory_pool.h
// Copyright (c) NVIDIA Corporation and affiliates.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMORY_POOL_MANAGED_MEMORY_POOL_HPP_
#define GCXX_RUNTIME_MEMORY_MEMORY_POOL_MANAGED_MEMORY_POOL_HPP_

#include <cstddef>
#include <utility>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/memory/buffers/no_init.hpp>
#include <gcxx/runtime/memory/buffers/properties.hpp>
#include <gcxx/runtime/memory/memory_resource/resource_concepts.hpp>
#include <gcxx/runtime/memory/mempool/memory_pool_properties.hpp>
#include <gcxx/runtime/memory/mempool/mempool_view.hpp>
#include <gcxx/runtime_backend/backend_stream_memory.hpp>

#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)

GCXX_NAMESPACE_MAIN_BEGIN()


class ManagedMemPoolView : public MemPoolView {
 public:
  // Managed memory is always host- and device-visible.
  using properties = TypeSet<device_accessible, host_accessible>;

  GCXX_FH explicit ManagedMemPoolView(driver::deviceMemPool_t pool) noexcept
      : MemPoolView(pool) {}

  ManagedMemPoolView(int)            = delete;
  ManagedMemPoolView(std::nullptr_t) = delete;
};

struct ManagedMemPool : ManagedMemPoolView {
  using reference_type = ManagedMemPoolView;

  GCXX_FH explicit ManagedMemPool(no_init_t) noexcept
      : ManagedMemPoolView(driver::deviceMemPool_t{}) {}

  GCXX_FH ManagedMemPool(memory_pool_properties props = {})
      : ManagedMemPoolView(create_memory_pool(
          flags::MemLocation::None, 0, flags::MemAllocation::Managed, props)) {}

  GCXX_FH ~ManagedMemPool() noexcept {
    if (m_pool_ != nullptr) {
      driver::deviceMemPoolDestroy(m_pool_);
    }
  }

  GCXX_FH static auto from_native_handle(driver::deviceMemPool_t pool) noexcept
    -> ManagedMemPool {
    return ManagedMemPool(pool);
  }

  // Hand-rolled instead of std::exchange: not constexpr until C++20.
  GCXX_FH constexpr auto release() noexcept -> driver::deviceMemPool_t {
    auto old = m_pool_;
    m_pool_  = nullptr;
    return old;
  }

  GCXX_FH auto as_ref() noexcept -> ManagedMemPoolView& {
    return static_cast<ManagedMemPoolView&>(*this);
  }

  ManagedMemPool(const ManagedMemPool&)                = delete;
  ManagedMemPool& operator=(const ManagedMemPool&)     = delete;
  ManagedMemPool(ManagedMemPool&&) noexcept            = delete;
  ManagedMemPool& operator=(ManagedMemPool&&) noexcept = delete;

 private:
  GCXX_FH explicit ManagedMemPool(driver::deviceMemPool_t pool) noexcept
      : ManagedMemPoolView(pool) {}
};

static_assert(
  resource_with<ManagedMemPoolView, device_accessible, host_accessible>,
  "ManagedMemPoolView must model the gcxx resource concept");
static_assert(resource_with<ManagedMemPool, device_accessible, host_accessible>,
              "ManagedMemPool must model the gcxx resource concept");


GCXX_NAMESPACE_MAIN_END()

#endif  // GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)

#endif
