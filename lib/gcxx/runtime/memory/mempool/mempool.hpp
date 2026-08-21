// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMPOOL_MEMPOOL_HPP_
#define GCXX_RUNTIME_MEMORY_MEMPOOL_MEMPOOL_HPP_

#include <utility>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/flags/memory_flags.hpp>
#include <gcxx/runtime/memory/mempool/memory_pool_properties.hpp>
#include <gcxx/runtime/memory/mempool/mempool_view.hpp>
#include <gcxx/runtime_backend/backend_device.hpp>
#include <gcxx/runtime_backend/backend_stream_memory.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


class MemPool : public MemPoolView {
 public:
  GCXX_FH explicit MemPool(
    flags::MemLocation locationType = flags::MemLocation::Device,
    int locationId                  = driver::deviceGet(),
    flags::MemAllocation allocType  = flags::MemAllocation::Pinned,
    memory_pool_properties props    = {})
      : MemPoolView(
          create_memory_pool(locationType, locationId, allocType, props)) {}

  GCXX_FH ~MemPool() noexcept {
    if (m_pool_ != nullptr) {
      driver::deviceMemPoolDestroy(m_pool_);
    }
  }

  MemPool(int)            = delete;
  MemPool(std::nullptr_t) = delete;

  MemPool(const MemPool&)            = delete;
  MemPool& operator=(const MemPool&) = delete;

  GCXX_FH MemPool(MemPool&& other) noexcept
      : MemPoolView(std::exchange(other.m_pool_, nullptr)) {}

  GCXX_FH auto operator=(MemPool&& other) noexcept -> MemPool& {
    if (this != &other) {
      if (m_pool_ != nullptr) {
        driver::deviceMemPoolDestroy(m_pool_);
      }
      m_pool_ = std::exchange(other.m_pool_, nullptr);
    }
    return *this;
  }

  GCXX_FH static auto from_native_handle(driver::deviceMemPool_t pool) noexcept
    -> MemPool {
    return MemPool(pool);
  }

  GCXX_FH auto Release() noexcept -> MemPoolView {
    auto pool = m_pool_;
    m_pool_   = nullptr;
    return MemPoolView{pool};
  }

 private:
  // Wrap an existing handle without creating a pool (for from_native_handle).
  GCXX_FH explicit MemPool(driver::deviceMemPool_t pool) noexcept
      : MemPoolView(pool) {}
};

GCXX_NAMESPACE_MAIN_END()

#endif
