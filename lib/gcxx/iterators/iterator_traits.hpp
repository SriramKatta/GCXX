// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_ITERATORS_ITERATOR_TRAITS_HPP_
#define GCXX_ITERATORS_ITERATOR_TRAITS_HPP_

#include <cstddef>
#include <iterator>

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

// std::iterator_traits, so gcxx iterators never name std:: directly.
template <typename Iterator_t>
using iterator_traits = std::iterator_traits<Iterator_t>;

// Aliases of the std tags so gcxx iterators never name std:: directly.
using input_iterator_tag         = std::input_iterator_tag;
using output_iterator_tag        = std::output_iterator_tag;
using forward_iterator_tag       = std::forward_iterator_tag;
using bidirectional_iterator_tag = std::bidirectional_iterator_tag;
using random_access_iterator_tag = std::random_access_iterator_tag;
#if defined(__cpp_lib_contiguous_iterator) && \
  __cpp_lib_contiguous_iterator >= 201902L
using contiguous_iterator_tag = std::contiguous_iterator_tag;
#endif


GCXX_NAMESPACE_MAIN_END()

#endif
