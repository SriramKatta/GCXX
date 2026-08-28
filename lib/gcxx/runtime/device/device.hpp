// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DEVICE_DEVICE_HPP_
#define GCXX_RUNTIME_DEVICE_DEVICE_HPP_

#include <gcxx/internal/prologue.hpp>


#include <gcxx/runtime/device/device_attributes.hpp>
#include <gcxx/runtime_backend/backend_device.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class DeviceHandle;

namespace Device {
  GCXX_FH auto set(device_t devId,
                   bool resetOnDestruct = false) -> DeviceHandle;

  GCXX_FH auto get() -> DeviceHandle;

  GCXX_FH auto count() -> int;

  // Non-throwing probe: true iff at least one device is visible. Unlike
  // count(), which aborts when no device is present (exceptions off).
  GCXX_FH auto available() -> bool;

  GCXX_FH auto sync() -> void;

  GCXX_FH auto getDeviceProp() -> driver::deviceProp_t;

  // Read a typed device attribute (see dev_attr).
  template <typename Attr>
  GCXX_FH auto attribute(Attr attr) -> typename Attr::type;

  // Read a typed device limit for the current device (see device_limits).
  template <typename Lim>
  GCXX_FH auto limit(Lim lim) -> typename Lim::type;

  // Write a typed device limit for the current device (see device_limits).
  template <typename Lim>
  GCXX_FH auto set_limit(Lim lim, typename Lim::type value) -> void;

};  // namespace Device

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/device/device.inl>

#endif