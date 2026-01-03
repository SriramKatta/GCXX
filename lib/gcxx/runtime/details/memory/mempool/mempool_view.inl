#pragma once
#ifndef GCXX_RUNTIME_DETAILS_MEMEORY_MEMPOOL_MEMPOOL_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_MEMEORY_MEMPOOL_MEMPOOL_VIEW_INL_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/device/device_handle.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FH auto MemPoolView::GetDefaultMempool(const DeviceHandle& hand)
  -> MemPoolView {
    deviceMemPool_t pool;
  GCXX_SAFE_RUNTIME_CALL(DeviceGetDefaultMemPool,
                         "failed to get the default memory pool for device",
                         &pool, hand.id());
  return {pool};
}

GCXX_NAMESPACE_MAIN_END


#endif