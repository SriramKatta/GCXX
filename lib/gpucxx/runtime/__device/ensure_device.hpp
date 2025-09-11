#pragma once

#ifndef GPUCXX_RUNTIME_DEVICE_ENSURE_DEVICE_HPP_
#define GPUCXX_RUNTIME_DEVICE_ENSURE_DEVICE_HPP_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>


GPUCXX_DETAILS_BEGIN_NAMESPACE

class [[maybe_unused]] __EnsureCurrentDevice {
 private:
  int old_device_{};
  bool changed_{false};

 public:
  // Constructor: takes a CUDA stream
  __EnsureCurrentDevice(int);

  // Destructor: restore old device if changed
  ~__EnsureCurrentDevice();

  // Delete copy constructor/assignment
  __EnsureCurrentDevice(const __EnsureCurrentDevice&)             = delete;
  __EnsureCurrentDevice& operator=(const __EnsureCurrentDevice&)  = delete;
  __EnsureCurrentDevice(const __EnsureCurrentDevice&&)            = delete;
  __EnsureCurrentDevice& operator=(const __EnsureCurrentDevice&&) = delete;
};

GPUCXX_DETAILS_END_NAMESPACE

#include <gpucxx/runtime/__details/ensure_device.inl>


#endif