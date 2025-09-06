#pragma once

#ifndef GPUCXX_RUNTIME_DETAILS_ENSURE_DEVICE_INL_
#define GPUCXX_RUNTIME_DETAILS_ENSURE_DEVICE_INL_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>
#include <gpucxx/runtime/runtime_error.hpp>


GPUCXX_DETAILS_BEGIN_NAMESPACE

// Constructor: takes a CUDA stream
__EnsureCurrentDevice::__EnsureCurrentDevice(int new_dev) {
  // Get current device
  GPUCXX_SAFE_RUNTIME_CALL(GetDevice, (&old_device_));

  changed_ = (old_device_ != new_dev);

  GPUCXX_SAFE_RUNTIME_CALL(SetDevice, (new_dev));
}

// Destructor: restore old device if changed
__EnsureCurrentDevice::~__EnsureCurrentDevice() {
  if (changed_) {
    GPUCXX_SAFE_RUNTIME_CALL(SetDevice, (old_device_));
  }
}

GPUCXX_DETAILS_END_NAMESPACE


#endif