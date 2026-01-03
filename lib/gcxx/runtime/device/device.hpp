#pragma once
#ifndef GCXX_RUNTIME_DEVICE_DEVICE_HPP_
#define GCXX_RUNTIME_DEVICE_DEVICE_HPP_

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

class DeviceHandle;

namespace Device {
  GCXX_FH auto set(device_t devId, bool resetOnDestruct = false)
    -> DeviceHandle;

  GCXX_FH auto get() -> DeviceHandle;

  GCXX_FH auto count() -> int;

  GCXX_FH auto Synchronize() -> void;
};  // namespace Device

GCXX_NAMESPACE_MAIN_END

#include <gcxx/runtime/details/device/device.inl>

#endif