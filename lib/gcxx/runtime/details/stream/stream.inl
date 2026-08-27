// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_STREAM_STREAM_INL_
#define GCXX_RUNTIME_DETAILS_STREAM_STREAM_INL_

#include <gcxx/runtime/device/ensure_current_device.hpp>
#include <gcxx/runtime/stream.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FH Stream::Stream(const flags::streamType createFlag,
                       const flags::streamPriority priorityFlag)
    : StreamView(driver::NULL_STREAM) {
  if (createFlag == flags::streamType::NullStream) {
    return;
  }
  m_stream = driver::streamCreateWithPriority(
    static_cast<details_::flag_t>(createFlag), -static_cast<int>(priorityFlag));
}

GCXX_FH auto Stream::operator=(Stream&& other) GCXX_NOEXCEPT -> Stream& {
  if (this != &other) {
    this->destroy();
    this->m_stream = std::exchange(other.m_stream, driver::INVALID_STREAM);
  }
  return *this;
}

GCXX_FH auto Stream::destroy() -> void {
  // since cudaStreamDestroy releases the handle after all work is done, to keep
  // similar behaviour
#if GCXX_HIP_MODE()
  if (!isInvalidStream()) {
    sync();
  }
#endif

  if (isNullStream() || isInvalidStream()) {
    return;
  }

  // NOTE : seems that this is not needed; we dont need to set the device back
  // to the device dtream is created on before destroying the stream
  driver::streamDestroy(m_stream);
  m_stream = driver::INVALID_STREAM;
}

GCXX_FH Stream::~Stream() {
  this->destroy();
}

GCXX_FH auto Stream::release() GCXX_NOEXCEPT -> StreamView {
  auto oldStream = m_stream;
  m_stream       = driver::INVALID_STREAM;
  return StreamView(oldStream);
}

GCXX_FH Stream::Stream(Stream&& other) noexcept
    : StreamView(std::exchange(other.m_stream, driver::INVALID_STREAM)) {}

GCXX_FH constexpr auto Stream::get() GCXX_CONST_NOEXCEPT -> StreamView {
  return *this;
}

GCXX_NAMESPACE_MAIN_END()


#endif