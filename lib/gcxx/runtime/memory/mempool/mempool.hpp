// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMPOOL_MEMPOOL_HPP_
#define GCXX_RUNTIME_MEMORY_MEMPOOL_MEMPOOL_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/memory/mempool/mempool_props.hpp>
#include <gcxx/runtime/memory/mempool/mempool_view.hpp>


GCXX_NAMESPACE_MAIN_BEGIN()

class MemPool : public MemPoolView {
 public:
  GCXX_FH explicit MemPool(const MemPoolProps&);

  GCXX_FH ~MemPool();

  MemPool(const MemPool&)            = delete;
  MemPool& operator=(const MemPool&) = delete;

  MemPool(std::nullptr_t) = delete;
  MemPool(int)            = delete;

  GCXX_FH MemPool(MemPool&& other) GCXX_NOEXCEPT;
  GCXX_FH auto operator=(MemPool&& other) GCXX_NOEXCEPT->MemPool&;

  GCXX_FH auto destroy() -> void;

  GCXX_FH auto Release() GCXX_NOEXCEPT -> MemPoolView;

  GCXX_FH constexpr auto get() GCXX_CONST_NOEXCEPT -> MemPoolView;
};

GCXX_NAMESPACE_MAIN_END()


#include <gcxx/runtime/details/memory/mempool/mempool.inl>

#endif