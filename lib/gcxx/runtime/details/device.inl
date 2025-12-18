#pragma once
#ifndef GCXX_RUNTIME_DETAILS_DEVICE_INL_
#define GCXX_RUNTIME_DETAILS_DEVICE_INL_


#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

#include <gcxx/runtime/device/device_handle.hpp>
#include <gcxx/runtime/device/device.hpp>


GCXX_NAMESPACE_MAIN_BEGIN


GCXX_FH auto Device::set(device_t devId, bool resetondestrcut ) -> DeviceHandle {
  return DeviceHandle(devId, resetondestrcut);
}

GCXX_FH auto Device::count() -> int {
  int num_dev;
  GCXX_SAFE_RUNTIME_CALL(GetDeviceCount, "Failed to Get device count",
                         &num_dev);
  return num_dev;
}

GCXX_NAMESPACE_MAIN_END

#endif