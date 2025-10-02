#pragma once

#ifndef GCXX_RUNTIME_DETAILS_ENSURE_DEVICE_INL_
#define GCXX_RUNTIME_DETAILS_ENSURE_DEVICE_INL_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/runtime_error.hpp>


GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

// Constructor: takes a CUDA stream
EnsureCurrentDevice::EnsureCurrentDevice(int new_dev) {
  // Get current device
  GCXX_SAFE_RUNTIME_CALL(GetDevice, (&old_device_));

  changed_ = (old_device_ != new_dev);

  GCXX_SAFE_RUNTIME_CALL(SetDevice, (new_dev));
}

// Destructor: restore old device if changed
EnsureCurrentDevice::~EnsureCurrentDevice() {
  if (changed_) {
    GCXX_SAFE_RUNTIME_CALL(SetDevice, (old_device_));
  }
}

GCXX_NAMESPACE_MAIN_DETAILS_END


#endif