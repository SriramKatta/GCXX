#pragma once
#ifndef GCXX_RUNTIME_DETAILS_DEVICE_DEVICE_INL_
#define GCXX_RUNTIME_DETAILS_DEVICE_DEVICE_INL_


#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/device/device_handle.hpp>
#include <gcxx/runtime/memory/mempool/mempool_view.hpp>


GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FH auto Device::set(device_t devId, bool resetOnDestruct) -> DeviceHandle {
  return DeviceHandle(devId, resetOnDestruct);
}

GCXX_FH auto Device::get() -> DeviceHandle {
  int dev_Id{};
  GCXX_SAFE_RUNTIME_CALL(GetDevice, "Failed to get device Id", &dev_Id);
  return DeviceHandle(dev_Id);
}

GCXX_FH auto Device::count() -> int {
  int num_dev{};
  GCXX_SAFE_RUNTIME_CALL(GetDeviceCount, "Failed to Get device count",
                         &num_dev);
  return num_dev;
}

GCXX_FH auto Device::Synchronize() -> void {
  GCXX_SAFE_RUNTIME_CALL(DeviceSynchronize, "Failed to synchronize the device");
}

GCXX_FH auto Device::getDeviceProp() -> DeviceProp {
  auto deviceId_ = get().id();
  DeviceProp handle;
  GCXX_SAFE_RUNTIME_CALL(GetDeviceProperties,
                         "Failed to query device properties", &handle,
                         deviceId_);
  return handle;
}

GCXX_FH auto Device::getAttribute(const flags::deviceAttribute& attr) -> int {
  auto deviceId_ = get().id();
  int val{};
  GCXX_SAFE_RUNTIME_CALL(DeviceGetAttribute, "Failed to query device attribute",
                         &val, static_cast<ATTRIBUTE_BACKEND_TYPE>(attr),
                         deviceId_);
  return val;
}

GCXX_FH auto Device::getLimit(const flags::deviceLimit& limattr)
  -> std::size_t {
  std::size_t pval{};
  GCXX_SAFE_RUNTIME_CALL(DeviceGetLimit, "Failed to get the device limit",
                         &pval, static_cast<LIMIT_BACKEND_TYPE>(limattr));
  return pval;
}

GCXX_FH auto Device::setLimit(const flags::deviceLimit& limattr,
                              std::size_t limval) -> void {
  GCXX_SAFE_RUNTIME_CALL(DeviceSetLimit, "Failed to set the device limit",
                         static_cast<LIMIT_BACKEND_TYPE>(limattr), limval);
}

GCXX_FH auto Device::GetDefaultMemPool() -> MemPoolView {
  auto deviceId_ = get().id();
  deviceMemPool_t pool{};
  GCXX_SAFE_RUNTIME_CALL(DeviceGetDefaultMemPool,
                         "Failed to get the defalt mempool of  the device",
                         &pool, deviceId_);
  return {pool};
}

GCXX_NAMESPACE_MAIN_END

#endif