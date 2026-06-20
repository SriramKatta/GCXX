// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_MEMORY_MEMPOOL_MEMPOOL_INL_
#define GCXX_RUNTIME_DETAILS_MEMORY_MEMPOOL_MEMPOOL_INL_

#include <gcxx/internal/prologue.hpp>

#include <utility>

#include <gcxx/runtime/memory/mempool/mempool.hpp>
#include <gcxx/runtime_backend/backend_stream_memory.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FH MemPool::MemPool(const MemPoolProps& props) {
  auto vals = props.getRawMemPoolProps();
  m_pool    = driver::deviceMemPoolCreate(vals);
}

GCXX_FH MemPool::MemPool(MemPool&& other) GCXX_NOEXCEPT
    : MemPoolView(std::exchange(other.m_pool, nullptr)) {}

GCXX_FH auto MemPool::operator=(MemPool&& other) GCXX_NOEXCEPT -> MemPool& {
  if (this != &other) {
    destroy();
    m_pool = std::exchange(other.m_pool, nullptr);
  }
  return *this;
}

GCXX_FH auto MemPool::destroy() -> void {
  if (m_pool != nullptr) {
    driver::deviceMemPoolDestroy(m_pool);
  }
  m_pool = nullptr;
}

GCXX_FH MemPool::~MemPool() {
  destroy();
}

GCXX_FH auto MemPool::Release() GCXX_NOEXCEPT -> MemPoolView {
  auto oldPool = m_pool;
  m_pool       = nullptr;
  return MemPoolView(oldPool);
}

GCXX_FH constexpr auto MemPool::get() GCXX_CONST_NOEXCEPT -> MemPoolView {
  return *this;
}

GCXX_NAMESPACE_MAIN_END()


#endif