#pragma once
#ifndef GPUCXX_API_RUNTIME_VECTOR_HPP_
#define GPUCXX_API_RUNTIME_VECTOR_HPP_

#include <gpucxx/runtime/__vector/vector_base.hpp>
#include <gpucxx/macros/define_macros.hpp>
#include <iterator>

GPUCXX_BEGIN_NAMESPACE

template <typename VT, typename Allocator_t>
class Vector : protected vector_base {
  
};

GPUCXX_END_NAMESPACE


#endif
