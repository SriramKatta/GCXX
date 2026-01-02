#pragma once
#ifndef GCXX_RUNTIME_DEVICE_DEVICE_HANDLE_HPP_
#define GCXX_RUNTIME_DEVICE_DEVICE_HANDLE_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>


#include <gcxx/runtime/device/device_structs.hpp>
#include <gcxx/runtime/flags/device_flags.hpp>


GCXX_NAMESPACE_MAIN_BEGIN

class DeviceHandle {
 private:
  const device_t deviceId_;
  const bool resetOnDestruct_;

 public:
  DeviceHandle() = delete;

  GCXX_FH explicit DeviceHandle(int dev, bool resetOnDestruct = false);

  GCXX_FH ~DeviceHandle();

  

  GCXX_FH auto makeCurrent() const -> void;

  GCXX_FH auto Synchronize() const -> void;

  GCXX_FH auto getDeviceProp() const -> DeviceProp;

  GCXX_FH auto getAttribute(const flags::deviceAttribute&) const -> int;

  GCXX_FH auto getLimit(const flags::deviceLimit&) const -> std::size_t;

  GCXX_FH auto setLimit(const flags::deviceLimit&, std::size_t) const -> void;

  GCXX_FHC auto id() const -> device_t;
};

GCXX_NAMESPACE_MAIN_END

#include <gcxx/runtime/details/device/device_handle.inl>

#endif