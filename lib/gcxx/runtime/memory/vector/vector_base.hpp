#pragma once
#ifndef GCXX_RUNTIME_VECTOR_VECTOR_BASE_HPP_
#define GCXX_RUNTIME_VECTOR_VECTOR_BASE_HPP_

#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/memory/vector/linear_storage.hpp>
#include <iterator>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

template <typename VT, typename allocator>
class vector_base {
  using m_storage = details_::linear_storage<VT, allocator>;
};

GCXX_NAMESPACE_MAIN_DETAILS_END


#endif
