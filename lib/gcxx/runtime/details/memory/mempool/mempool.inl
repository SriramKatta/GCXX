// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_MEMEORY_MEMPOOL_MEMPOOL_INL_
#define GCXX_RUNTIME_DETAILS_MEMEORY_MEMPOOL_MEMPOOL_INL_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_stream_memory.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FH MemPool::MemPool(const MemPoolProps& props) {
  auto vals = props.getRawMemPoolProps();
  m_pool    = driver::deviceMemPoolCreate(vals);
}

GCXX_FH auto MemPool::destroy() -> void {
  if (m_pool) {
    driver::deviceMemPoolDestroy(m_pool);
  }
  m_pool = nullptr;
}

GCXX_FH MemPool::~MemPool() {
  destroy();
}

GCXX_NAMESPACE_MAIN_END()


#endif