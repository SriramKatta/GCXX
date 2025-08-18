#pragma once
#ifndef GPUCXX_API_DETAILS_RUNTIME_EVENT_INL_
#define GPUCXX_API_DETAILS_RUNTIME_EVENT_INL_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/runtime/__flags/eventflags.hpp>
#include <gpucxx/runtime/runtime_error.hpp>
#include <gpucxx/macros/define_macros.hpp>

#include <utility>

GPUCXX_BEGIN_NAMESPACE

Event::Event(const flags::eventCreate createFlag) {
  GPUCXX_SAFE_RUNTIME_CALL(EventCreateWithFlags,
                           (&event_, static_cast<flag_t>(createFlag)));
}

auto Event::destroy() -> void {
  if (event_) {
    GPUCXX_SAFE_RUNTIME_CALL(EventDestroy, (event_));
    event_ = nullptr;
  }
}

Event::~Event() {
  this->destroy();
}

auto Event::query() const -> bool {
  auto err            = GPUCXX_RUNTIME_BACKEND(EventQuery)(event_);
  constexpr auto line = __LINE__ - 1;
  switch (err) {
    case GPUCXX_RUNTIME_BACKEND(Success):
      return true;
    case GPUCXX_RUNTIME_BACKEND(ErrorNotReady):
      return false;
    default:
      details_::checkDeviceError(err, "event Query", __FILE__, line);
      return false;
  }
}

auto Event::RecordInStream(const deviceStream_t &str,
                           const flags::eventRecord recordFlag) -> void {
  GPUCXX_SAFE_RUNTIME_CALL(EventRecordWithFlags,
                           (event_, str, static_cast<flag_t>(recordFlag)));
}

auto Event::Synchronize() const -> void {
  GPUCXX_SAFE_RUNTIME_CALL(EventSynchronize, (event_));
}

Event::Event(Event &&other) noexcept
    : event_(std::exchange(other.event_, nullptr)) {}

Event::operator deviceRawEvent_t() const {
  return event_;
}

auto Event::release() noexcept -> deviceRawEvent_t {
  return std::exchange(event_, nullptr);
}

auto Event::operator=(Event &&other) noexcept -> Event & {
  if (this != &other) {
    this->destroy();
    event_ = std::exchange(other.event_, nullptr);
  }
  return *this;
}

auto Event::ElapsedTimeSince(const Event &start) const -> float {
  this->Synchronize();
  float ms{};
  GPUCXX_SAFE_RUNTIME_CALL(EventElapsedTime, (&ms, start.event_, this->event_));
  return ms;
}

auto ElapsedTime(const Event &start, const Event &stop) -> float {
  return stop.ElapsedTimeSince(start);
}

auto Event::Create(const flags::eventCreate createFlag) -> Event {
  return {createFlag};
}

GPUCXX_END_NAMESPACE


#endif
