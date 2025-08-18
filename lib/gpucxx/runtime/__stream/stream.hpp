#pragma once
#ifndef GPUCXX_API_RUNTIME_STREAM_HPP_
#define GPUCXX_API_RUNTIME_STREAM_HPP_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/runtime/flags/eventflags.hpp>
#include <gpucxx/runtime/flags/streamflags.hpp>
#include <gpucxx/runtime/runtime_error.hpp>
#include <gpucxx/macros/define_macros.hpp>

#include <cstddef>
#include <utility>

GPUCXX_BEGIN_NAMESPACE

class Stream : stream_ref{
 private:
  GPUCXX_FH Stream(const flags::streamBehaviour createFlag,
                   const flags::streamPriority priorityFlag);

  GPUCXX_FH auto destroy() -> void;

 public:
  GPUCXX_FH static auto Create(
    const flags::streamBehaviour = flags::streamBehaviour::Null,
    const flags::streamPriority  = flags::streamPriority::Default) -> Stream;

  GPUCXX_FH ~Stream();


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


GPUCXX_END_NAMESPACE

#include <gpucxx/details/runtime/stream.inl>

#endif