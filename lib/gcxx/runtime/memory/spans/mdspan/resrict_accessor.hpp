#pragma once
#ifndef GCXX_RUNTIME_MEMORY_SPANS_MDSPAN_MDSPAN_HPP
#define GCXX_RUNTIME_MEMORY_SPANS_MDSPAN_MDSPAN_HPP

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

template <class ElementType>
struct restrict_accessor {
  using offset_policy    = gcxx::default_accessor<ElementType>;
  using element_type     = ElementType;
  using reference        = ElementType&;
  using data_handle_type = ElementType*;

  constexpr restrict_accessor() noexcept = default;

  template <class OtherElementType,
            typename std::enable_if_t<
              std::is_convertible_v<OtherElementType (*)[], element_type (*)[]>,
              int> = 0>
  constexpr restrict_accessor(restrict_accessor<OtherElementType>) noexcept {}

  constexpr reference access(data_handle_type p, size_t i) const noexcept {
    return p[i];
  }

  constexpr typename offset_policy::data_handle_type offset(
    data_handle_type p, size_t i) const noexcept {
    return p + i;
  }
};

GCXX_NAMESPACE_MAIN_END


#endif