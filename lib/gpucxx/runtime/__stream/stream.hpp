#pragma once
#ifndef GPUCXX_API_RUNTIME_STREAM_STREAM_HPP_
#define GPUCXX_API_RUNTIME_STREAM_STREAM_HPP_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>
#include <gpucxx/runtime/__flags/eventflags.hpp>
#include <gpucxx/runtime/__flags/streamflags.hpp>

#include <cstddef>
#include <utility>

GPUCXX_BEGIN_NAMESPACE

class Stream : public stream_ref {
 public:
  GPUCXX_FH Stream(
    const flags::streamType createFlag       = flags::streamType::defaultStream,
    const flags::streamPriority priorityFlag = flags::streamPriority::none);

  GPUCXX_FH static auto Create(
    const flags::streamType createFlag       = flags::streamType::defaultStream,
    const flags::streamPriority priorityFlag = flags::streamPriority::none)
    -> Stream;

  GPUCXX_FH ~Stream();

  Stream(int) = delete;

  Stream(std::nullptr_t) = delete;

  Stream(const Stream&) = delete;

  Stream& operator=(const Stream&) = delete;

  Stream(Stream&& other) noexcept;

  GPUCXX_FH auto operator=(Stream&& other) GPUCXX_NOEXCEPT->Stream&;


  GPUCXX_FH auto release() GPUCXX_NOEXCEPT -> stream_ref;


  GPUCXX_FH auto getPriority() -> flags::streamPriority;

 private:
  GPUCXX_FH auto destroy() -> void;
};

GPUCXX_END_NAMESPACE

#include <gpucxx/runtime/__details/stream.inl>

#include <gpucxx/macros/undefine_macros.hpp>

#endif
