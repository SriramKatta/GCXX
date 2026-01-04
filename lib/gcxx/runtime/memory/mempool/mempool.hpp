#pragma once
#ifndef GCXX_RUNTIME_MEMEORY_MEMPOOL_MEMPOOL_HPP_
#define GCXX_RUNTIME_MEMEORY_MEMPOOL_MEMPOOL_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/memory/mempool/mempool_props.hpp>
#include <gcxx/runtime/memory/mempool/mempool_view.hpp>


GCXX_NAMESPACE_MAIN_BEGIN

class MemPool : public MemPoolView {
 public:
  GCXX_FH MemPool(const MemPoolProps&);

  GCXX_FH auto destroy() -> void;

  GCXX_FH ~MemPool();
};

GCXX_NAMESPACE_MAIN_END


#include <gcxx/runtime/details/memory/mempool/mempool.inl>

#endif