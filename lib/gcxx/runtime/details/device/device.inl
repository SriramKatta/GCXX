// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_DEVICE_DEVICE_INL_
#define GCXX_RUNTIME_DETAILS_DEVICE_DEVICE_INL_


#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/device/device_handle.hpp>


GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FH
auto Device::set(device_t devId, bool resetOnDestruct) -> DeviceHandle {
  return DeviceHandle(devId, resetOnDestruct);
}

GCXX_FH auto Device::get() -> DeviceHandle {
  auto dev_Id = driver::deviceGet();
  return DeviceHandle(dev_Id);
}

GCXX_FH auto Device::count() -> int {
  auto num_dev = driver::deviceGetCount();
  return num_dev;
}

GCXX_FH auto Device::Synchronize() -> void {
  driver::deviceSynchronize();
}

GCXX_FH auto Device::getDeviceProp() -> driver::deviceProp_t {
  auto deviceId_ = get().id();
  auto handle    = driver::deviceGetProp(deviceId_);
  return handle;
}

template <typename Attr>
GCXX_FH auto Device::attribute(Attr attr) -> typename Attr::type {
  return attr(get().id());
}

template <typename Lim>
GCXX_FH auto Device::limit(Lim lim) -> typename Lim::type {
  return lim();
}

template <typename Lim>
GCXX_FH auto Device::set_limit(Lim /*lim*/, typename Lim::type value) -> void {
  Lim::set(value);
}

GCXX_NAMESPACE_MAIN_END()

#endif