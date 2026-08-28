// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_DEVICE_DEVICE_HANDLE_INL_
#define GCXX_RUNTIME_DETAILS_DEVICE_DEVICE_HANDLE_INL_


#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/device/ensure_current_device.hpp>
#include <gcxx/runtime/flags/device_flags.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FH DeviceHandle::DeviceHandle(int devId, bool resetOnDestruct)
    : m_deviceId(devId), m_resetOnDestruct(resetOnDestruct) {
  makeCurrent();
}

GCXX_FH DeviceHandle::~DeviceHandle() {
  if (m_resetOnDestruct) {
    [[maybe_unused]] const details_::EnsureCurrentDevice hand(m_deviceId);
    driver::deviceReset();
  }
}

GCXX_FH auto DeviceHandle::makeCurrent() const -> void {
  driver::deviceSet(m_deviceId);
}

GCXX_FH auto DeviceHandle::sync() const -> void {
  [[maybe_unused]] const details_::EnsureCurrentDevice hand(m_deviceId);
  gcxx::Device::sync();
}

template <typename Attr>
GCXX_FH auto DeviceHandle::attribute(Attr attr) const -> typename Attr::type {
  return attr(m_deviceId);
}

GCXX_FHC auto DeviceHandle::id() const -> device_t {
  return m_deviceId;
}

template <typename Lim>
GCXX_FH auto DeviceHandle::limit(Lim lim) const -> typename Lim::type {
  // Limits operate on the current device, so make this handle's device current.
  [[maybe_unused]] const details_::EnsureCurrentDevice dev(m_deviceId);
  return lim();
}

template <typename Lim>
GCXX_FH auto DeviceHandle::set_limit(Lim /*lim*/,
                                     typename Lim::type value) const -> void {
  [[maybe_unused]] const details_::EnsureCurrentDevice dev(m_deviceId);
  Lim::set(value);
}

GCXX_FH auto DeviceHandle::getDeviceProp() const -> driver::deviceProp_t {
  [[maybe_unused]] const details_::EnsureCurrentDevice dev(m_deviceId);
  return gcxx::Device::getDeviceProp();
}

GCXX_NAMESPACE_MAIN_END()

#endif