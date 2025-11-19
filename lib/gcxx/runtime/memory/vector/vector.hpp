#pragma once
#ifndef GCXX_RUNTIME_VECTOR_VECTOR_HPP_
#define GCXX_RUNTIME_VECTOR_VECTOR_HPP_

#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/details/vector/vector_base.hpp>
#include <iterator>

GCXX_NAMESPACE_MAIN_BEGIN

template <typename VT, typename allocator>
class Vector : protected details_::vector_base<VT, allocator> {};

GCXX_NAMESPACE_MAIN_END


#endif
