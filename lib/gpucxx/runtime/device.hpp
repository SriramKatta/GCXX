#pragma once
#ifndef GPUCXX_RUNTIME_DEVICE_HPP_
#define GPUCXX_RUNTIME_DEVICE_HPP_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>

GPUCXX_BEGIN_NAMESPACE

class DeviceRef {
 private:
  using device_t = int;
  device_t deviceId_;
  bool resetOnDestrcut_;

 public:
  GPUCXX_FH explicit DeviceRef(int devId, bool resetondestrcut = false)
      : deviceId_(devId), resetOnDestrcut_(resetondestrcut) {
    GPUCXX_SAFE_RUNTIME_CALL(SetDevice, (devId));
  }

  GPUCXX_FH ~DeviceRef() {
    if (resetOnDestrcut_) {
      GPUCXX_SAFE_RUNTIME_CALL(DeviceReset, ());
    }
  }

  GPUCXX_FH auto Synchronize() const -> void {
    GPUCXX_SAFE_RUNTIME_CALL(DeviceSynchronize, ());
  }

  DeviceRef(const DeviceRef &)                     = delete;
  DeviceRef(const DeviceRef &&)                    = delete;
  DeviceRef &operator=(const DeviceRef &)          = delete;
  DeviceRef &operator=(DeviceRef &&other) noexcept = delete;

  GPUCXX_FH auto id() const -> device_t { return deviceId_; }
};

GPUCXX_END_NAMESPACE

#endif