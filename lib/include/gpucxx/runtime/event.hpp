#pragma once
#ifndef GPUCXX_API_RUNTIME_EVENT_HPP
#define GPUCXX_API_RUNTIME_EVENT_HPP

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/runtime/flags/eventflags.hpp>
#include <gpucxx/utils/define_specifiers.hpp>


GPUCXX_BEGIN_NAMESPACE

class Event {
 private:
  using deviceRawEvent_t = GPUCXX_RUNTIME_BACKEND(Event_t);
  using deviceStream_t   = GPUCXX_RUNTIME_BACKEND(Stream_t);
  deviceRawEvent_t event_{};

  GPUCXX_FH Event(const flags::eventCreate createFlag);

  GPUCXX_FH auto destroy() -> void;

 public:
  GPUCXX_FH friend auto EventCreate(const flags::eventCreate) -> Event;

  GPUCXX_FH ~Event();

  GPUCXX_FH auto query() const -> bool;

  GPUCXX_FH auto RecordInStream(
    const deviceStream_t &str           = nullptr,
    const flags::eventRecord recordFlag = flags::eventRecord::Default) -> void;

  GPUCXX_FH auto Synchronize() const -> void;

  Event(const Event &)            = delete;
  Event &operator=(const Event &) = delete;

  Event(Event &&other) noexcept;

  GPUCXX_FH operator deviceRawEvent_t() const;

  GPUCXX_FH deviceRawEvent_t release() noexcept;

  GPUCXX_FH auto operator=(Event &&other) noexcept -> Event &;

  GPUCXX_FH auto ElapsedTimeSince(const Event &start) const -> float;
};

GPUCXX_FH auto ElapsedTime(const Event &start, const Event &stop) -> float;

[[nodiscard]] GPUCXX_FH auto EventCreate(
  const flags::eventCreate createFlag = flags::eventCreate::Default) -> Event;

GPUCXX_END_NAMESPACE

#include <gpucxx/details/runtime/event.inl>

#endif
