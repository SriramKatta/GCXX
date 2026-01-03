#pragma once
#ifndef GCXX_RUNTIME_MEMEORY_MEMPOOL_MEMPOOL_VIEW_HPP_
#define GCXX_RUNTIME_MEMEORY_MEMPOOL_MEMPOOL_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/device/device_handle.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

using deviceMemPool_t = GCXX_RUNTIME_BACKEND(MemPool_t);

class MemPoolView {
 protected:
  deviceMemPool_t pool_;

 public:
  MemPoolView(deviceMemPool_t pool) : pool_(pool) {}

  GCXX_FH auto getRawMemPool() const -> deviceMemPool_t;

  GCXX_FH static auto GetDefaultMempool(const DeviceHandle&) -> MemPoolView;
};

GCXX_NAMESPACE_MAIN_END


#include <gcxx/runtime/details/memory/mempool/mempool_view.inl>

#endif