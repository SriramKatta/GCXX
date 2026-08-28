// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DEVICE_ENSURE_DEVICE_HPP_
#define GCXX_RUNTIME_DEVICE_ENSURE_DEVICE_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_device.hpp>


GCXX_NAMESPACE_MAIN_DETAILS_BEGIN()

class [[maybe_unused]] EnsureCurrentDevice {

 public:
  GCXX_FH EnsureCurrentDevice(device_t);

  // Destructor: restore old device if changed.
  GCXX_FH ~EnsureCurrentDevice();

  EnsureCurrentDevice(const EnsureCurrentDevice&)             = delete;
  EnsureCurrentDevice& operator=(const EnsureCurrentDevice&)  = delete;
  EnsureCurrentDevice(const EnsureCurrentDevice&&)            = delete;
  EnsureCurrentDevice& operator=(const EnsureCurrentDevice&&) = delete;

 private:
  int m_old_device{};
  bool m_changed{false};
};

GCXX_NAMESPACE_MAIN_DETAILS_END()

#include <gcxx/runtime/details/device/ensure_device.inl>


#endif