// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_API_RUNTIME_STREAM_STREAM_HPP_
#define GCXX_API_RUNTIME_STREAM_STREAM_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/flags/event_flags.hpp>
#include <gcxx/runtime/flags/stream_flags.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

#include <cstddef>
#include <utility>

GCXX_NAMESPACE_MAIN_BEGIN()

class Stream : public StreamView {
  GCXX_FH auto destroy() -> void;

 public:
  GCXX_FH explicit Stream(
    flags::streamType createFlag       = flags::streamType::SyncWithNull,
    flags::streamPriority priorityFlag = flags::streamPriority::None);

  GCXX_FH ~Stream();

  Stream(int) = delete;

  Stream(std::nullptr_t) = delete;

  Stream(const Stream&) = delete;

  Stream& operator=(const Stream&) = delete;

  GCXX_FH Stream(Stream&& other) noexcept;

  GCXX_FH auto operator=(Stream&& other) GCXX_NOEXCEPT->Stream&;

  GCXX_FH constexpr auto get() GCXX_CONST_NOEXCEPT -> StreamView;

  GCXX_FH auto Release() GCXX_NOEXCEPT -> StreamView;

  GCXX_FH auto getPriority() -> flags::streamPriority;
};

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/stream/stream.inl>

#include <gcxx/macros/undefine_macros.hpp>

#endif
