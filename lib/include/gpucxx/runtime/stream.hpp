#pragma once
#ifndef GPUCXX_API_RUNTIME_STREAM_HPP
#define GPUCXX_API_RUNTIME_STREAM_HPP

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/runtime/runtime_error.hpp>
#include <gpucxx/runtime/flags/eventflags.hpp>
#include <gpucxx/runtime/flags/streamflags.hpp>
#include <gpucxx/utils/define_specifiers.hpp>

#include <cstddef>
#include <utility>

GPUCXX_BEGIN_NAMESPACE

class Stream {
 private:
  using deviceRawEvent_t  = GPUCXX_RUNTIME_BACKEND(Event_t);
  using deviceRawStream_t = GPUCXX_RUNTIME_BACKEND(Stream_t);
  deviceRawStream_t stream_{nullptr};

  GPUCXX_FH Stream(const flags::streamBehaviour createFlag,
                   const flags::streamPriority priorityFlag);

  GPUCXX_FH auto destroy() -> void;

 public:
  GPUCXX_FH friend auto StreamCreate(const flags::streamBehaviour,
                                     const flags::streamPriority) -> Stream;

  GPUCXX_FH ~Stream();

  GPUCXX_FH auto query() const -> bool;

  GPUCXX_FH auto Synchronize() const -> void;

  GPUCXX_FH auto WaitOnEvent(
    const deviceRawEvent_t &event,
    const flags::eventWait waitFlag = flags::eventWait::Default) const -> void;

  Stream(int)                       = delete;
  Stream(std::nullptr_t)            = delete;
  Stream(const Stream &)            = delete;
  Stream &operator=(const Stream &) = delete;

  Stream(Stream &&other) noexcept;

  GPUCXX_FH operator deviceRawStream_t() const;

  GPUCXX_FH deviceRawStream_t release() noexcept;
  GPUCXX_FH auto operator=(Stream &&other) noexcept -> Stream &;

  GPUCXX_FH auto getPriority() -> flags::streamPriority;
};

GPUCXX_FH auto StreamCreate(
  const flags::streamBehaviour createFlag  = flags::streamBehaviour::Null,
  const flags::streamPriority priorityFlag = flags::streamPriority::default) -> Stream;

GPUCXX_END_NAMESPACE

#include <gpucxx/details/runtime/stream.inl>

#endif