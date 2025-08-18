#pragma once
#ifndef GPUCXX_API_RUNTIME_EVENT_EVENT_REF_HPP_
#define GPUCXX_API_RUNTIME_EVENT_EVENT_REF_HPP_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>
#include <gpucxx/runtime/__flags/eventflags.hpp>


GPUCXX_BEGIN_NAMESPACE

class event_ref {
 protected:
  using deviceRawEvent_t = GPUCXX_RUNTIME_BACKEND(Event_t);
  using deviceStream_t   = GPUCXX_RUNTIME_BACKEND(Stream_t);

 public:
  GPUCXX_FH auto query() const -> bool;

  GPUCXX_FH auto RecordInStream(
    const deviceStream_t& str           = nullptr,
    const flags::eventRecord recordFlag = flags::eventRecord::none) -> void;

  GPUCXX_FH auto Synchronize() const -> void;

  Event(const Event&)            = delete;
  Event& operator=(const Event&) = delete;

  Event(Event&& other) noexcept;

  GPUCXX_FH operator deviceRawEvent_t() const;

  GPUCXX_FH deviceRawEvent_t release() noexcept;

  GPUCXX_FH auto operator=(Event&& other) noexcept -> Event&;

  GPUCXX_FH auto ElapsedTimeSince(const Event& start) const -> float;

 protected:
  deviceRawEvent_t event_{};
};

GPUCXX_FH auto ElapsedTime(const Event& start, const Event& stop) -> float;

GPUCXX_END_NAMESPACE

#include <gpucxx/runtime/__details/event.inl>

#endif
