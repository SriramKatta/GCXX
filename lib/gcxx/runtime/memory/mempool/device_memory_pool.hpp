// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Derived from NVIDIA CCCL libcudacxx:
//   https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__memory_resource/__memory_pool/DeviceMemPool.h
// Copyright (c) NVIDIA Corporation and affiliates.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMORY_POOL_DeviceMemPool_HPP_
#define GCXX_RUNTIME_MEMORY_MEMORY_POOL_DeviceMemPool_HPP_

#include <cstddef>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/device/device_handle.hpp>
#include <gcxx/runtime/memory/buffers/no_init.hpp>
#include <gcxx/runtime/memory/buffers/properties.hpp>
#include <gcxx/runtime/memory/memory_resource/resource_concepts.hpp>
#include <gcxx/runtime/memory/mempool/memory_pool_properties.hpp>
#include <gcxx/runtime/memory/mempool/mempool_view.hpp>
#include <gcxx/runtime_backend/backend_stream_memory.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


class DeviceMemPoolView : public MemPoolView {
 public:
  // Pool memory is always device-visible.
  using properties = TypeSet<device_accessible>;

  GCXX_FH explicit DeviceMemPoolView(driver::deviceMemPool_t pool) noexcept
      : MemPoolView(pool) {}

  DeviceMemPoolView(int)            = delete;
  DeviceMemPoolView(std::nullptr_t) = delete;
};

struct DeviceMemPool : DeviceMemPoolView {
  using reference_type = DeviceMemPoolView;

  GCXX_FH explicit DeviceMemPool(no_init_t) noexcept
      : DeviceMemPoolView(driver::deviceMemPool_t{}) {}

  GCXX_FH explicit DeviceMemPool(const gcxx::DeviceHandle& device,
                                 memory_pool_properties props = {})
      : DeviceMemPoolView(
          create_memory_pool(flags::MemLocation::Device, device.id(),
                             flags::MemAllocation::Pinned, props)) {}

  GCXX_FH ~DeviceMemPool() noexcept {
    if (m_pool_ != nullptr) {
      driver::deviceMemPoolDestroy(m_pool_);
    }
  }

  GCXX_FH static auto from_native_handle(driver::deviceMemPool_t pool) noexcept
    -> DeviceMemPool {
    return DeviceMemPool(pool);
  }

  // Hand-rolled instead of std::exchange: not constexpr until C++20.
  GCXX_FH constexpr auto release() noexcept -> driver::deviceMemPool_t {
    auto old = m_pool_;
    m_pool_  = nullptr;
    return old;
  }

  GCXX_FH auto as_ref() noexcept -> DeviceMemPoolView& {
    return static_cast<DeviceMemPoolView&>(*this);
  }

  DeviceMemPool(const DeviceMemPool&)                = delete;
  DeviceMemPool& operator=(const DeviceMemPool&)     = delete;
  DeviceMemPool(DeviceMemPool&&) noexcept            = delete;
  DeviceMemPool& operator=(DeviceMemPool&&) noexcept = delete;

 private:
  GCXX_FH explicit DeviceMemPool(driver::deviceMemPool_t pool) noexcept
      : DeviceMemPoolView(pool) {}
};

static_assert(resource_with<DeviceMemPoolView, device_accessible>,
              "DeviceMemPoolView must model the gcxx resource concept");
static_assert(resource_with<DeviceMemPool, device_accessible>,
              "DeviceMemPool must model the gcxx resource concept");


GCXX_NAMESPACE_MAIN_END()

#endif
