// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_BUFFERS_BUFFER_HPP_
#define GCXX_RUNTIME_MEMORY_BUFFERS_BUFFER_HPP_

#include <type_traits>

#include <gcxx/internal/prologue.hpp>


// GCXX_NAMESPACE_DETAILS_BEGIN()

// template <typename resource, bool = std::is_empty_v<R> &&
// !std::is_final_v<R>> struct resource_storage;


// template <typename resource>
// struct resource_storage<resource, false> : private R {
//     // todo add some rule 5
// };


// template <typename resource>
// struct resource_storage<resource, true> {
//     // todo add some rule 5
//   resource m_r{};
// };

// GCXX_NAMESPACE_DETAILS_END()


GCXX_NAMESPACE_MAIN_BEGIN()

template <typename VT, typename resource>
class unintialised_buffer : private resource {


  GCXX_TEMPLATE()
  GCXX_REQUIRES(std::is_constructible_v<resource>)
  explicit unintialised_buffer(std::size_t N)
      : m_data(resource::allocate(N)), m_size(N) {
    GCXX_STATIC_EXPECT("")
  }

 private:
  VT* m_data{};
  std::size_t m_size{};
};

GCXX_NAMESPACE_MAIN_END()

#endif