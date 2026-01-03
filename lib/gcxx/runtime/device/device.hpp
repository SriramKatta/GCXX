#pragma once
#ifndef GCXX_RUNTIME_DEVICE_DEVICE_HPP_
#define GCXX_RUNTIME_DEVICE_DEVICE_HPP_

#include <gcxx/internal/prologue.hpp>


#include <gcxx/runtime/device/device_structs.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

class DeviceHandle;
class MemPoolView;

namespace Device {
  GCXX_FH auto set(device_t devId, bool resetOnDestruct = false)
    -> DeviceHandle;

  GCXX_FH auto get() -> DeviceHandle;

  GCXX_FH auto count() -> int;

  GCXX_FH auto Synchronize() -> void;

  GCXX_FH auto getDeviceProp() -> DeviceProp;

  GCXX_FH auto getAttribute(const flags::deviceAttribute&) -> int;

  GCXX_FH auto getLimit(const flags::deviceLimit&) -> std::size_t;

  GCXX_FH auto setLimit(const flags::deviceLimit&, std::size_t) -> void;

  GCXX_FH auto GetDefaultMemPool() -> MemPoolView;

};  // namespace Device

GCXX_NAMESPACE_MAIN_END

#include <gcxx/runtime/details/device/device.inl>

#endif