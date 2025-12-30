#pragma once
#ifndef GCXX_RUNTIME_DETAILS_DEVICE_DEVICE_INL_
#define GCXX_RUNTIME_DETAILS_DEVICE_DEVICE_INL_


#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

#include <gcxx/runtime/device/device.hpp>
#include <gcxx/runtime/device/device_handle.hpp>


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

GCXX_NAMESPACE_MAIN_END

#endif