#pragma once
#ifndef GCXX_RUNTIME_STREAM_STREAM_REF_HPP_
#define GCXX_RUNTIME_STREAM_STREAM_REF_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/event/event_base.hpp>
#include <gcxx/runtime/flags/eventflags.hpp>
#include <gcxx/runtime/flags/streamflags.hpp>

#include <gcxx/runtime/event/event.hpp>


GCXX_DETAILS_BEGIN_NAMESPACE
// clang-format off
using deviceStream_t                    = GCXX_RUNTIME_BACKEND(Stream_t);
inline static const auto NULL_STREAM    = reinterpret_cast<deviceStream_t>(0ULL); // NOLINT
inline static const auto INVALID_STREAM = reinterpret_cast<deviceStream_t>(~0ULL); // NOLINT
// clang-format on
GCXX_DETAILS_END_NAMESPACE


GCXX_BEGIN_NAMESPACE
class Event;

class stream_ref {
 protected:
  using deviceStream_t = GCXX_RUNTIME_BACKEND(Stream_t);

 public:
  constexpr stream_ref(deviceStream_t rawStream) noexcept
      : stream_(rawStream) {}

  stream_ref()               = delete;
  stream_ref(int)            = delete;
  stream_ref(std::nullptr_t) = delete;

  GCXX_FH constexpr auto get() GCXX_CONST_NOEXCEPT -> deviceStream_t {
    return stream_;
  }

  GCXX_FH constexpr operator deviceStream_t() GCXX_CONST_NOEXCEPT {
    return get();
  }

  GCXX_FH auto HasPendingWork() -> bool;

  GCXX_FH auto Synchronize() const -> void;

  GCXX_FH auto WaitOnEvent(
    const details_::event_ref& event,
    const flags::eventWait waitFlag = flags::eventWait::none) const -> void;

  GCXX_FH auto recordEvent(
    const flags::eventCreate createflag = flags::eventCreate::none,
    const flags::eventRecord recordFlag = flags::eventRecord::none) const
    -> Event;


 protected:
  deviceStream_t stream_{details_::NULL_STREAM};  // NOLINT
};

GCXX_END_NAMESPACE


#include <gcxx/macros/undefine_macros.hpp>
#include <gcxx/runtime/details/stream_ref.inl>

#endif