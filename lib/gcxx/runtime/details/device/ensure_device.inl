#pragma once

#ifndef GCXX_RUNTIME_DETAILS_DEVICE_ENSURE_DEVICE_INL_
#define GCXX_RUNTIME_DETAILS_DEVICE_ENSURE_DEVICE_INL_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/runtime_error.hpp>


GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

// Constructor: takes a CUDA stream
GCXX_FH EnsureCurrentDevice::EnsureCurrentDevice(device_t new_dev) {
  // Get current device
  GCXX_SAFE_RUNTIME_CALL(GetDevice, "Failed to get current GPU ID",
                         &old_device_);

  changed_ = (old_device_ != new_dev);

  GCXX_SAFE_RUNTIME_CALL(SetDevice, "Failed to Set Device", new_dev);
}

// Destructor: restore old device if changed
GCXX_FH EnsureCurrentDevice::~EnsureCurrentDevice() {
  if (changed_) {
    GCXX_SAFE_RUNTIME_CALL(SetDevice, "Failed to Reset GPU ID", old_device_);
  }
}

GCXX_NAMESPACE_MAIN_DETAILS_END


#endif