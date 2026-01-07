#pragma once
#ifndef GCXX_RUNTIME_MEMORY_SPANS_MDSPAN_MDSPAN_HPP
#define GCXX_RUNTIME_MEMORY_SPANS_MDSPAN_MDSPAN_HPP

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

template <class Accessor>
struct restrict_accessor : public Accessor
{
public:
    using offset_policy = restrict_accessor<typename Accessor::offset_policy>;
    using element_type = typename Accessor::element_type;
    using reference = typename Accessor::reference;
    using data_handle_type = element_type * GCXX_RESTRICT_KEYWORD;
    constexpr reference access(data_handle_type p, std::size_t i) const noexcept{
        return Accessor::access(p, i);
    }
    
    constexpr data_handle_type offset( data_handle_type p, std::size_t i ) const noexcept{
        return Accessor::offset(p, i);
    }

};

GCXX_NAMESPACE_MAIN_END


#endif