#pragma once
#ifndef GPUCXX_RUNTIME_STREAM_STREAM_REF_HPP_
#define GPUCXX_RUNTIME_STREAM_STREAM_REF_HPP_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>
#include <gpucxx/runtime/__flags/eventflags.hpp>
#include <gpucxx/runtime/__flags/streamflags.hpp>
#include <gpucxx/runtime/__event/event_ref.hpp>
#include <gpucxx/runtime/runtime_error.hpp>

GPUCXX_DETAILS_BEGIN_NAMESPACE

using deviceStream_t = GPUCXX_RUNTIME_BACKEND(Stream_t);
static const deviceStream_t __null_stream =
  reinterpret_cast<deviceStream_t>(0ULL);
static const deviceStream_t __invalid_stream =
  reinterpret_cast<deviceStream_t>(~0ULL);

GPUCXX_DETAILS_END_NAMESPACE


GPUCXX_BEGIN_NAMESPACE

class stream_ref {
 protected:
  using deviceStream_t = GPUCXX_RUNTIME_BACKEND(Stream_t);

 public:
  constexpr stream_ref(deviceStream_t __str) noexcept : stream_(__str) {}

  stream_ref()               = default;
  stream_ref(int)            = delete;
  stream_ref(std::nullptr_t) = delete;

  GPUCXX_FHD constexpr auto get() const noexcept -> deviceStream_t {
    return stream_;
  }

  GPUCXX_FH auto HasPendingWork() -> bool;

  GPUCXX_FH auto Synchronize() const -> void;

  GPUCXX_FH auto WaitOnEvent(
    const event_ref &event,
    const flags::eventWait waitFlag = flags::eventWait::none) const -> void;

 protected:
  deviceStream_t stream_{details_::__invalid_stream};
};

GPUCXX_END_NAMESPACE


#include <gpucxx/runtime/__details/stream_ref.inl>

#endif