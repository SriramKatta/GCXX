#pragma once
#ifndef GCXX_RUNTIME_DETAILS_DEVICE_HANDLE_INL_
#define GCXX_RUNTIME_DETAILS_DEVICE_HANDLE_INL_


#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

#include <gcxx/runtime/device/ensure_current_device.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FH DeviceHandle::DeviceHandle(int devId, bool resetondestrcut)
    : deviceId_(devId), resetOnDestrcut_(resetondestrcut) {
  GCXX_SAFE_RUNTIME_CALL(SetDevice, "Failed to Set device", devId);
}

GCXX_FH DeviceHandle::~DeviceHandle() {
  if (resetOnDestrcut_) {
    GCXX_SAFE_RUNTIME_CALL(DeviceReset, "Failed to reset ");
  }
}

GCXX_FH auto DeviceHandle::Synchronize() const -> void {
  details_::EnsureCurrentDevice hand(deviceId_);
  GCXX_SAFE_RUNTIME_CALL(DeviceSynchronize, "Failed to synchronize the device");
}

GCXX_FH auto DeviceHandle::id() const -> device_t {
  return deviceId_;
}

GCXX_NAMESPACE_MAIN_END

#endif