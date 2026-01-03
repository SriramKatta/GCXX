#pragma once
#ifndef GCXX_RUNTIME_DETAILS_DEVICE_DEVICE_HANDLE_INL_
#define GCXX_RUNTIME_DETAILS_DEVICE_DEVICE_HANDLE_INL_


#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/device/ensure_current_device.hpp>
#include <gcxx/runtime/flags/device_flags.hpp>
#include <gcxx/runtime/memory/mempool/mempool_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

GCXX_FH DeviceHandle::DeviceHandle(int devId, bool resetOnDestruct)
    : deviceId_(devId), resetOnDestruct_(resetOnDestruct) {
  makeCurrent();
}

GCXX_FH DeviceHandle::~DeviceHandle() {
  if (resetOnDestruct_) {
    details_::EnsureCurrentDevice hand(deviceId_);
    GCXX_SAFE_RUNTIME_CALL(DeviceReset, "Failed to reset ");
  }
}

GCXX_FH auto DeviceHandle::makeCurrent() const -> void {
  GCXX_SAFE_RUNTIME_CALL(SetDevice, "Failed to Set device", deviceId_);
}

GCXX_FH auto DeviceHandle::Synchronize() const -> void {
  details_::EnsureCurrentDevice hand(deviceId_);
  gcxx::Device::Synchronize();
}

GCXX_FH auto DeviceHandle::getAttribute(
  const flags::deviceAttribute& attr) const -> int {
  details_::EnsureCurrentDevice dev(deviceId_);
  return gcxx::Device::getAttribute(attr);
}

GCXX_FHC auto DeviceHandle::id() const -> device_t {
  return deviceId_;
}

GCXX_FH auto DeviceHandle::getLimit(const flags::deviceLimit& limattr) const
  -> std::size_t {
  details_::EnsureCurrentDevice dev(deviceId_);
  return gcxx::Device::getLimit(limattr);
}

GCXX_FH auto DeviceHandle::setLimit(const flags::deviceLimit& limattr,
                                    std::size_t limval) const -> void {
  details_::EnsureCurrentDevice dev(deviceId_);
  gcxx::Device::setLimit(limattr, limval);
}

GCXX_FH auto DeviceHandle::getDeviceProp() const -> DeviceProp {
  details_::EnsureCurrentDevice dev(deviceId_);
  return gcxx::Device::getDeviceProp();
}

GCXX_FH auto DeviceHandle::GetDefaultMemPool() const -> MemPoolView {
  details_::EnsureCurrentDevice dev(deviceId_);
  return gcxx::Device::GetDefaultMemPool();
}

GCXX_NAMESPACE_MAIN_END

#endif