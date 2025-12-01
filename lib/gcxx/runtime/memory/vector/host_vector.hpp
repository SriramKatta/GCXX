#pragma once
#ifndef GCXX_RUNTIME_VECTOR_HOST_VECTOR_HPP_
#define GCXX_RUNTIME_VECTOR_HOST_VECTOR_HPP_

#include <vector>

#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/memory/allocators/host_allocator.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

template<typename VT>
using host_vector = std::vector<VT, gcxx::details_::host_allocator<VT>>;

GCXX_NAMESPACE_MAIN_END


#endif
