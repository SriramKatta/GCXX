// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DEVICE_DEVICE_HANDLE_HPP_
#define GCXX_RUNTIME_DEVICE_DEVICE_HANDLE_HPP_

#include <gcxx/internal/prologue.hpp>


#include <gcxx/runtime/flags/device_flags.hpp>
#include <gcxx/runtime_backend/backend_device.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class MemPoolView;

class DeviceHandle {


 public:
  DeviceHandle() = delete;

  GCXX_FH explicit DeviceHandle(int dev, bool resetOnDestruct = false);

  GCXX_FH ~DeviceHandle();

  GCXX_FH auto makeCurrent() const -> void;

  GCXX_FH auto Synchronize() const -> void;

  GCXX_FH auto getDeviceProp() const -> driver::deviceProp_t;

  GCXX_FH auto getAttribute(const flags::deviceAttribute&) const -> int;

  GCXX_FH auto getLimit(const flags::deviceLimit&) const -> std::size_t;

  GCXX_FH auto setLimit(const flags::deviceLimit&, std::size_t) const -> void;

  GCXX_FHC auto id() const -> device_t;

  GCXX_FH auto GetDefaultMemPool() const -> MemPoolView;

  GCXX_FH auto SetMemPool(const MemPoolView&) -> void;

  GCXX_FH auto GetMemPool() -> MemPoolView;

 private:
  const device_t m_deviceId;
  const bool m_resetOnDestruct;
};

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/device/device_handle.inl>

#endif