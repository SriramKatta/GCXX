#pragma once
#ifndef GPUCXX_API_RUNTIME_EVENT_HPP
#define GPUCXX_API_RUNTIME_EVENT_HPP

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/error/runtime_error.hpp>
#include <gpucxx/runtime/flags/eventflags.hpp>
#include <gpucxx/utils/define_specifiers.hpp>

#include <utility>

GPUCXX_BEGIN_NAMESPACE

class Event {
 private:
  using deviceRawEvent = GPUCXX_RUNTIME_BACKEND(Event_t);
  using deviceStream_t = GPUCXX_RUNTIME_BACKEND(Stream_t);
  deviceRawEvent event_{};

  GPUCXX_FH Event(const flags::eventCreate flag) {
    GPUCXX_SAFE_RUNTIME_CALL(EventCreateWithFlags,
                             (&event_, static_cast<flag_t>(flag)));
  }

  GPUCXX_FH auto destroy() noexcept -> void {
    if (event_) {
      GPUCXX_SAFE_RUNTIME_CALL(EventDestroy, (event_));
      event_ = nullptr;
    }
  }

 public:
  GPUCXX_FH friend auto EventCreate(const flags::eventCreate) -> Event;

  GPUCXX_FH ~Event() { this->destroy(); }

  GPUCXX_FH auto query() const -> bool {
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

  GPUCXX_FH auto RecordInStream(
    const deviceStream_t &str     = nullptr,
    const flags::eventRecord flag = flags::eventRecord::Default) -> void {
    GPUCXX_SAFE_RUNTIME_CALL(EventRecordWithFlags,
                             (event_, str, static_cast<flag_t>(flag)));
  }

  GPUCXX_FH auto Synchronize() const -> void {
    GPUCXX_SAFE_RUNTIME_CALL(EventSynchronize, (event_));
  }

  Event(const Event &)            = delete;
  Event &operator=(const Event &) = delete;

  Event(Event &&other) noexcept : event_(other.event_) {
    other.event_ = nullptr;
  }

  GPUCXX_FH operator deviceRawEvent() const { return event_; }

  GPUCXX_FH deviceRawEvent release() { return std::exchange(event_, nullptr); }

  GPUCXX_FH auto operator=(Event &&other) noexcept -> Event & {
    if (this != &other) {
      this->destroy();
      event_ = std::exchange(other.event_, nullptr);
    }
    return *this;
  }

  GPUCXX_FH auto ElapsedTimeSince(const Event &start) const -> float {
    this->Synchronize();
    float ms{};
    GPUCXX_SAFE_RUNTIME_CALL(EventElapsedTime,
                             (&ms, start.event_, this->event_));
    return ms;
  }
};

[[nodiscard]] GPUCXX_FH auto EventCreate(
  const flags::eventCreate flag = flags::eventCreate::Default) -> Event {
  return {flag};
}

GPUCXX_FH auto ElapsedTime(const Event &start, const Event &stop) -> float {
  return stop.ElapsedTimeSince(start);
}

GPUCXX_END_NAMESPACE

#endif
