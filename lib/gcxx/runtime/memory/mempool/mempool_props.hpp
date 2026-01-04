#pragma once
#ifndef GCXX_RUNTIME_MEMEORY_MEMPOOL_MEMPOOL_PROPS_HPP_
#define GCXX_RUNTIME_MEMEORY_MEMPOOL_MEMPOOL_PROPS_HPP_

#include <gcxx/internal/prologue.hpp>


#include <gcxx/runtime/flags/memory_flags.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

using deviceMemPoolProps_t = GCXX_RUNTIME_BACKEND(MemPoolProps);

struct MemPoolProps {


  GCXX_FH auto getRawMemPoolProps() -> deviceMemPoolProps_t {
    deviceMemPoolProps_t props{};
    std::memset(&props, 0, sizeof(props));
    props.allocType = 0;
  }
};

GCXX_NAMESPACE_MAIN_END


#endif