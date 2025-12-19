#pragma once
#ifndef GCXX_RUNTIME_DETAILS_DEVICE_HANDLE_INL_
#define GCXX_RUNTIME_DETAILS_DEVICE_HANDLE_INL_


#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

#include <gcxx/runtime/device/ensure_current_device.hpp>
#include <gcxx/runtime/flags/device_flags.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FH DeviceHandle::DeviceHandle(int devId, bool resetondestrcut)
    : deviceId_(devId), resetOnDestrcut_(resetondestrcut) {
  makeCurrent();
}

GCXX_FH DeviceHandle::~DeviceHandle() {
  if (resetOnDestrcut_) {
    details_::EnsureCurrentDevice hand(deviceId_);
    GCXX_SAFE_RUNTIME_CALL(DeviceReset, "Failed to reset ");
  }
}

GCXX_FH auto DeviceHandle::makeCurrent() const -> void {
  GCXX_SAFE_RUNTIME_CALL(SetDevice, "Failed to Set device", deviceId_);
}

GCXX_FH auto DeviceHandle::Synchronize() const -> void {
  details_::EnsureCurrentDevice hand(deviceId_);
  GCXX_SAFE_RUNTIME_CALL(DeviceSynchronize, "Failed to synchronize the device");
}

GCXX_FH auto DeviceHandle::getAttribute(
  const flags::deviceAttribute& attr) const -> int {
  details_::EnsureCurrentDevice dev(deviceId_);
  int val{};
  GCXX_SAFE_RUNTIME_CALL(DeviceGetAttribute,
                         "Failed to query device attaribute", &val,
                         static_cast<ATTRIBUTE_BACKEND_TYPE>(attr), deviceId_);
  return val;
}

GCXX_FHC auto DeviceHandle::id() const -> device_t {
  return deviceId_;
}

GCXX_FH auto DeviceHandle::getLimit(const flags::deviceLimit& limattr) const
  -> std::size_t {
  details_::EnsureCurrentDevice dev(deviceId_);
  std::size_t pval{};
  GCXX_SAFE_RUNTIME_CALL(DeviceGetLimit, "Failed to get the device limit",
                         &pval, static_cast<LIMIT_BACKEND_TYPE>(limattr));
  return pval;
}

GCXX_FH auto DeviceHandle::setLimit(const flags::deviceLimit& limattr,
                                    std::size_t limval) const -> void {
  details_::EnsureCurrentDevice dev(deviceId_);
  GCXX_SAFE_RUNTIME_CALL(DeviceSetLimit, "Failed to set the device limit",
                         static_cast<LIMIT_BACKEND_TYPE>(limattr), limval);
}

GCXX_FH auto DeviceHandle::getDeviceProp() const -> DeviceProp {
  DeviceProp handle;
  GCXX_SAFE_RUNTIME_CALL(GetDeviceProperties,
                         "Failed to query device properties", &handle,
                         deviceId_);
  return handle;
}

GCXX_NAMESPACE_MAIN_END

#endif