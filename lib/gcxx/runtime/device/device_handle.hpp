#pragma once
#ifndef GCXX_RUNTIME_DEVICE_DEVICE_HANDLE_HPP_
#define GCXX_RUNTIME_DEVICE_DEVICE_HANDLE_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>


GCXX_NAMESPACE_MAIN_BEGIN

class DeviceHandle {
 private:
  device_t deviceId_;
  bool resetOnDestrcut_;

 public:
  GCXX_FH explicit DeviceHandle(int devId, bool resetondestrcut = false);

  GCXX_FH ~DeviceHandle();

  GCXX_FH auto Synchronize() const -> void;

  GCXX_FH auto id() const -> device_t;

  GCXX_FH static auto set(device_t devId) -> void;

  GCXX_FH static auto count() -> int;
};

GCXX_NAMESPACE_MAIN_END

#include <gcxx/runtime/details/device_handle.inl>

#endif