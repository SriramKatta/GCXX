#pragma once

#ifndef GCXX_RUNTIME_DEVICE_ENSURE_DEVICE_HPP_
#define GCXX_RUNTIME_DEVICE_ENSURE_DEVICE_HPP_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>


GCXX_DETAILS_BEGIN_NAMESPACE

class [[maybe_unused]] EnsureCurrentDevice {
 private:
  int old_device_{};
  bool changed_{false};

 public:
  GCXX_FH EnsureCurrentDevice(int);

  // Destructor: restore old device if changed
  GCXX_FH ~EnsureCurrentDevice();

  // Delete copy constructor/assignment
  EnsureCurrentDevice(const EnsureCurrentDevice&)             = delete;
  EnsureCurrentDevice& operator=(const EnsureCurrentDevice&)  = delete;
  EnsureCurrentDevice(const EnsureCurrentDevice&&)            = delete;
  EnsureCurrentDevice& operator=(const EnsureCurrentDevice&&) = delete;
};

GCXX_DETAILS_END_NAMESPACE

#include <gpucxx/runtime/details/ensure_device.inl>


#endif