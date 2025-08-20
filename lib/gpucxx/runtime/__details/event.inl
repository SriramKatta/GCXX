#pragma once
#ifndef GPUCXX_RUNTIME_DETAILS_EVENT_INL_
#define GPUCXX_RUNTIME_DETAILS_EVENT_INL_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>
#include <gpucxx/runtime/__flags/eventflags.hpp>
#include <gpucxx/runtime/runtime_error.hpp>

#include <utility>

GPUCXX_BEGIN_NAMESPACE

GPUCXX_FH Event::Event(const flags::eventCreate createFlag) {
  GPUCXX_SAFE_RUNTIME_CALL(EventCreateWithFlags,
                           (&event_, static_cast<flag_t>(createFlag)));
}

GPUCXX_FH Event::~Event() {
  GPUCXX_SAFE_RUNTIME_CALL(EventDestroy, (event_));
}

GPUCXX_FH Event::Event(Event&& other) noexcept : event_ref(std::move(other)) {}

GPUCXX_END_NAMESPACE


#endif
