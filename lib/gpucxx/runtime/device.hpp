#pragma once
#ifndef GCXX_RUNTIME_DEVICE_HPP_
#define GCXX_RUNTIME_DEVICE_HPP_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>

GCXX_BEGIN_NAMESPACE

class DeviceRef {
 private:
  using device_t = int;
  device_t deviceId_;
  bool resetOnDestrcut_;

 public:
  GCXX_FH explicit DeviceRef(int devId, bool resetondestrcut = false)
      : deviceId_(devId), resetOnDestrcut_(resetondestrcut) {
    GCXX_SAFE_RUNTIME_CALL(SetDevice, (devId));
  }

  GCXX_FH ~DeviceRef() {
    if (resetOnDestrcut_) {
      GCXX_SAFE_RUNTIME_CALL(DeviceReset, ());
    }
  }

  GCXX_FH auto Synchronize() const -> void {
    GCXX_SAFE_RUNTIME_CALL(DeviceSynchronize, ());
  }

  DeviceRef(const DeviceRef&)                      = delete;
  DeviceRef(const DeviceRef&&)                     = delete;
  DeviceRef& operator=(const DeviceRef&)           = delete;
  DeviceRef& operator=(DeviceRef&& other) noexcept = delete;

  GCXX_FH auto id() const -> device_t { return deviceId_; }
};

GCXX_END_NAMESPACE

#endif