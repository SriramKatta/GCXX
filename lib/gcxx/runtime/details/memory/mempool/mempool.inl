#pragma once
#ifndef GCXX_RUNTIME_DETAILS_MEMEORY_MEMPOOL_MEMPOOL_INL_
#define GCXX_RUNTIME_DETAILS_MEMEORY_MEMPOOL_MEMPOOL_INL_

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FH MemPool::MemPool(const MemPoolProps& props) {
  auto vals = props.getRawMemPoolProps();
  GCXX_SAFE_RUNTIME_CALL(MemPoolCreate, "failed to create memory pool", &pool_,
                         &vals);
}

GCXX_FH auto MemPool::destroy() -> void {
  if (pool_) {
    GCXX_SAFE_RUNTIME_CALL(MemPoolDestroy, "failed to destroy memory pool",
                           pool_);
  }
  pool_ = nullptr;
}

GCXX_FH MemPool::~MemPool() {
  destroy();
}

GCXX_NAMESPACE_MAIN_END


#endif