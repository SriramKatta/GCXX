#pragma once
#ifndef GPUCXX_RUNTIME_EVENT_EVENT_REF_HPP_
#define GPUCXX_RUNTIME_EVENT_EVENT_REF_HPP_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>
#include <gpucxx/runtime/__flags/eventflags.hpp>
#include <gpucxx/runtime/__stream/stream_ref.hpp>


GPUCXX_BEGIN_NAMESPACE

class event_ref : event_base{

 protected:
  using deviceEvent_t = GPUCXX_RUNTIME_BACKEND(Event_t);

 public:
  constexpr event_ref(deviceEvent_t __evt) noexcept : event_base(__evt) {}

  event_ref()               = delete;
  event_ref(int)            = delete;
  event_ref(std::nullptr_t) = delete;


  GPUCXX_FH auto HasOccurred() const -> bool;

  GPUCXX_FH auto Synchronize() const -> void;

  GPUCXX_FH auto RecordInStream(
    const stream_ref& stream      = details_::__null_stream,
    flags::eventRecord recordFlag = flags::eventRecord::none) -> void;


  GPUCXX_FH auto GetElapsedTime(const event_ref& startEvent) const -> double;

 protected:
  deviceEvent_t event_{static_cast<deviceEvent_t>(0ULL)};
};

GPUCXX_END_NAMESPACE

#include <gpucxx/runtime/__details/event_ref.inl>

#include <gpucxx/macros/undefine_macros.hpp>

#endif
