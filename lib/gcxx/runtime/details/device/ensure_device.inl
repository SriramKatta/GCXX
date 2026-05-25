// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once

#ifndef GCXX_RUNTIME_DETAILS_DEVICE_ENSURE_DEVICE_INL_
#define GCXX_RUNTIME_DETAILS_DEVICE_ENSURE_DEVICE_INL_

#include <gcxx/internal/prologue.hpp>


GCXX_NAMESPACE_MAIN_DETAILS_BEGIN()

// Constructor: takes a CUDA stream
GCXX_FH EnsureCurrentDevice::EnsureCurrentDevice(device_t new_dev) {
  // Get current device
  old_device_ = driver::deviceGet();
  changed_    = (old_device_ != new_dev);
  driver::deviceSet(new_dev);
}

// Destructor: restore old device if changed
GCXX_FH EnsureCurrentDevice::~EnsureCurrentDevice() {
  if (changed_) {
    driver::deviceSet(old_device_);
  }
}

GCXX_NAMESPACE_MAIN_DETAILS_END()


#endif