#pragma once
#ifndef GPUCXX_RUNTIME_STREAM_STREAM_REF_HPP_
#define GPUCXX_RUNTIME_STREAM_STREAM_REF_HPP_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>
#include <gpucxx/runtime/__event/event_base.hpp>
#include <gpucxx/runtime/__flags/eventflags.hpp>
#include <gpucxx/runtime/__flags/streamflags.hpp>

#include <gpucxx/runtime/__event/event.hpp>


GPUCXX_DETAILS_BEGIN_NAMESPACE
// clang-format off
using deviceStream_t                       = GPUCXX_RUNTIME_BACKEND(Stream_t);
inline static const auto __null_stream_    = reinterpret_cast<deviceStream_t>(0ULL); // NOLINT
inline static const auto __invalid_stream_ = reinterpret_cast<deviceStream_t>(~0ULL); // NOLINT
// clang-format on
GPUCXX_DETAILS_END_NAMESPACE


GPUCXX_BEGIN_NAMESPACE
class Event;

class stream_ref {
 protected:
  using deviceStream_t = GPUCXX_RUNTIME_BACKEND(Stream_t);

 public:
  constexpr stream_ref(deviceStream_t __str) noexcept : stream_(__str) {}

  stream_ref()               = delete;
  stream_ref(int)            = delete;
  stream_ref(std::nullptr_t) = delete;

  GPUCXX_FH constexpr auto get() GPUCXX_CONST_NOEXCEPT -> deviceStream_t {
    return stream_;
  }

  GPUCXX_FH constexpr operator deviceStream_t() GPUCXX_CONST_NOEXCEPT {
    return get();
  }

  GPUCXX_FH auto HasPendingWork() -> bool;

  GPUCXX_FH auto Synchronize() const -> void;

  GPUCXX_FH auto WaitOnEvent(
    const details_::event_ref& event,
    const flags::eventWait waitFlag = flags::eventWait::none) const -> void;

  GPUCXX_FH auto recordEvent(
    const flags::eventCreate createflag = flags::eventCreate::none,
    const flags::eventRecord recordFlag = flags::eventRecord::none) const
    -> Event;


 protected:
  deviceStream_t stream_{details_::__null_stream_};  // NOLINT
};

GPUCXX_END_NAMESPACE


#include <gpucxx/macros/undefine_macros.hpp>
#include <gpucxx/runtime/__details/stream_ref.inl>

#endif