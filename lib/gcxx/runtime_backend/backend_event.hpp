// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_EVENT_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_EVENT_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime_backend/backend_handles.hpp>

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN

GCXX_FH auto eventCreateWithFlags(details_::flag_t createFlag)
  -> deviceEvent_t {
  deviceEvent_t event{};
  GCXX_SAFE_RUNTIME_CALL(EventCreateWithFlags, "Failed to create GPU Event",
                         &event, createFlag);
  return event;
}

GCXX_FH auto eventDestroy(deviceEvent_t event) -> void {
  GCXX_SAFE_RUNTIME_CALL(EventDestroy, "Failed to destroy GPU Event", event);
}

GCXX_FH auto eventElapsedTime(deviceEvent_t startEvent,
                              deviceEvent_t endEvent) -> float {
  float ms{};
  GCXX_SAFE_RUNTIME_CALL(EventElapsedTime,
                         "Failed to get elapsed time between GPU Events", &ms,
                         startEvent, endEvent);
  return ms;
}

GCXX_FH auto eventQueryNoThrow(deviceEvent_t event) -> deviceError_t {
  return GCXX_RUNTIME_BACKEND(EventQuery)(event);
}

GCXX_FH auto eventRecordWithFlags(deviceEvent_t event, deviceStream_t stream,
                                  details_::flag_t recordFlag) -> void {
  GCXX_SAFE_RUNTIME_CALL(EventRecordWithFlags,
                         "Failed to record GPU Event in GPU Stream", event,
                         stream, recordFlag);
}

GCXX_FH auto eventSynchronize(deviceEvent_t event) -> void {
  GCXX_SAFE_RUNTIME_CALL(EventSynchronize, "Failed to synchronize GPU Event",
                         event);
}

GCXX_NAMESPACE_MAIN_DRIVER_END

#endif
