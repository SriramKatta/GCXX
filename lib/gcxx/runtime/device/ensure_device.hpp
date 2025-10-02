#pragma once

#ifndef GCXX_RUNTIME_DEVICE_ENSURE_DEVICE_HPP_
#define GCXX_RUNTIME_DEVICE_ENSURE_DEVICE_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>


GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

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

GCXX_NAMESPACE_MAIN_DETAILS_END

#include <gcxx/runtime/details/ensure_device.inl>


#endif