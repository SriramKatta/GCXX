// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_DEVICE_DEVICE_HANDLE_INL_
#define GCXX_RUNTIME_DETAILS_DEVICE_DEVICE_HANDLE_INL_


#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/device/ensure_current_device.hpp>
#include <gcxx/runtime/flags/device_flags.hpp>
#include <gcxx/runtime/memory/mempool/mempool_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FH DeviceHandle::DeviceHandle(int devId, bool resetOnDestruct)
    : m_deviceId(devId), m_resetOnDestruct(resetOnDestruct) {
  makeCurrent();
}

GCXX_FH DeviceHandle::~DeviceHandle() {
  if (m_resetOnDestruct) {
    details_::EnsureCurrentDevice hand(m_deviceId);
    driver::deviceReset();
  }
}

GCXX_FH auto DeviceHandle::makeCurrent() const -> void {
  driver::deviceSet(m_deviceId);
}

GCXX_FH auto DeviceHandle::Synchronize() const -> void {
  details_::EnsureCurrentDevice hand(m_deviceId);
  gcxx::Device::Synchronize();
}

GCXX_FH
auto DeviceHandle::getAttribute(const flags::deviceAttribute& attr) const
  -> int {
  details_::EnsureCurrentDevice dev(m_deviceId);
  return gcxx::Device::getAttribute(attr);
}

GCXX_FHC auto DeviceHandle::id() const -> device_t {
  return m_deviceId;
}

GCXX_FH
auto DeviceHandle::getLimit(const flags::deviceLimit& limattr) const
  -> std::size_t {
  details_::EnsureCurrentDevice dev(m_deviceId);
  return gcxx::Device::getLimit(limattr);
}

GCXX_FH
auto DeviceHandle::setLimit(const flags::deviceLimit& limattr,
                            std::size_t limval) const -> void {
  details_::EnsureCurrentDevice dev(m_deviceId);
  gcxx::Device::setLimit(limattr, limval);
}

GCXX_FH auto DeviceHandle::getDeviceProp() const -> driver::deviceProp_t {
  details_::EnsureCurrentDevice dev(m_deviceId);
  return gcxx::Device::getDeviceProp();
}

GCXX_FH auto DeviceHandle::GetDefaultMemPool() const -> MemPoolView {
  details_::EnsureCurrentDevice dev(m_deviceId);
  return gcxx::Device::GetDefaultMemPool();
}

GCXX_FH auto DeviceHandle::SetMemPool(const MemPoolView& pool) -> void {
  details_::EnsureCurrentDevice dev(m_deviceId);
  return gcxx::Device::SetMemPool(pool);
}

GCXX_FH auto DeviceHandle::GetMemPool() -> MemPoolView {
  details_::EnsureCurrentDevice dev(m_deviceId);
  return gcxx::Device::GetMemPool();
}

GCXX_NAMESPACE_MAIN_END()

#endif