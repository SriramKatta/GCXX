// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_DEVICE_DEVICE_INL_
#define GCXX_RUNTIME_DETAILS_DEVICE_DEVICE_INL_


#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/device/device_handle.hpp>
#include <gcxx/runtime/memory/mempool/mempool_view.hpp>


GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FH
auto Device::set(device_t devId, bool resetOnDestruct) -> DeviceHandle {
  return DeviceHandle(devId, resetOnDestruct);
}

GCXX_FH auto Device::get() -> DeviceHandle {
  auto dev_Id = driver::deviceGet();
  return DeviceHandle(dev_Id);
}

GCXX_FH auto Device::count() -> int {
  auto num_dev = driver::deviceGetCount();
  return num_dev;
}

GCXX_FH auto Device::Synchronize() -> void {
  driver::deviceSynchronize();
}

GCXX_FH auto Device::getDeviceProp() -> driver::deviceProp_t {
  auto deviceId_ = get().id();
  auto handle    = driver::deviceGetProp(deviceId_);
  return handle;
}

GCXX_FH auto Device::getAttribute(const flags::deviceAttribute& attr) -> int {
  auto deviceId_ = get().id();
  const auto val = driver::deviceGetAttribute(
    static_cast<ATTRIBUTE_BACKEND_TYPE>(attr), deviceId_);
  return val;
}

GCXX_FH
auto Device::getLimit(const flags::deviceLimit& limattr) -> std::size_t {
  std::size_t pval =
    driver::deviceGetLimit(static_cast<LIMIT_BACKEND_TYPE>(limattr));
  return pval;
}

GCXX_FH
auto Device::setLimit(const flags::deviceLimit& limattr,
                      std::size_t limval) -> void {
  driver::deviceSetLimit(static_cast<LIMIT_BACKEND_TYPE>(limattr), limval);
}

GCXX_FH auto Device::GetDefaultMemPool() -> MemPoolView {
  auto deviceId_ = get().id();
  auto pool      = driver::deviceGetDefaultMemoryPool(deviceId_);
  return {pool};
}

GCXX_FH auto Device::SetMemPool(const MemPoolView& pool) -> void {
  auto deviceId_ = get().id();
  driver::deviceSetMemPool(deviceId_, pool.getRawMemPool());
}

GCXX_FH auto Device::GetMemPool() -> MemPoolView {
  auto deviceId_ = get().id();
  auto pool      = driver::deviceGetMemPool(deviceId_);
  return {pool};
}

GCXX_NAMESPACE_MAIN_END()

#endif