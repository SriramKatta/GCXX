#pragma once
#ifndef GCXX_RUNTIME_DETAILS_MEMEORY_MEMPOOL_MEMPOOL_VIEW_INL_
#define GCXX_RUNTIME_DETAILS_MEMEORY_MEMPOOL_MEMPOOL_VIEW_INL_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/device/device_handle.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FH auto MemPoolView::getRawMemPool() const -> deviceMemPool_t {
  return pool_;
};

GCXX_FH auto MemPoolView::GetDefaultMempool(const DeviceHandle& hand)
  -> MemPoolView {
  return hand.GetDefaultMemPool();
}

GCXX_NAMESPACE_MAIN_END


#endif