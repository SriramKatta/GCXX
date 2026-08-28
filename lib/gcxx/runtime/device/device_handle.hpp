// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DEVICE_DEVICE_HANDLE_HPP_
#define GCXX_RUNTIME_DEVICE_DEVICE_HANDLE_HPP_

#include <gcxx/internal/prologue.hpp>


#include <gcxx/runtime/flags/device_flags.hpp>
#include <gcxx/runtime_backend/backend_device.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class DeviceHandle {


 public:
  DeviceHandle() = delete;

  GCXX_FH explicit DeviceHandle(int dev, bool resetOnDestruct = false);

  GCXX_FH ~DeviceHandle();

  GCXX_FH auto makeCurrent() const -> void;

  GCXX_FH auto sync() const -> void;

  GCXX_FH auto getDeviceProp() const -> driver::deviceProp_t;

  // Read a typed device attribute scoped to this handle's device.
  template <typename Attr>
  GCXX_FH auto attribute(Attr attr) const -> typename Attr::type;

  // Read a typed device limit (makes this handle's device current first).
  template <typename Lim>
  GCXX_FH auto limit(Lim lim) const -> typename Lim::type;

  // Write a typed device limit (makes this handle's device current first).
  template <typename Lim>
  GCXX_FH auto set_limit(Lim lim, typename Lim::type value) const -> void;

  GCXX_FHC auto id() const -> device_t;

  // RAII owner of the previous-device state: copying/moving could
  // double-restore.
  DeviceHandle(const DeviceHandle&)                    = delete;
  auto operator=(const DeviceHandle&) -> DeviceHandle& = delete;
  DeviceHandle(DeviceHandle&&)                         = delete;
  auto operator=(DeviceHandle&&) -> DeviceHandle&      = delete;

 private:
  device_t m_deviceId;
  bool m_resetOnDestruct;
};

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/device/device_handle.inl>

#endif