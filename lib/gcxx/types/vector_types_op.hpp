// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_TYPES_VECTOR_TYPES_OP_HPP
#define GCXX_TYPES_VECTOR_TYPES_OP_HPP

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/types/vector_types.hpp>
#include <type_traits>
#include <utility>


GCXX_NAMESPACE_MAIN_DETAILS_BEGIN()

// LHS and RHS are unrelated template parameters — an operation is vectorized
// when EITHER operand is a vector — so the two sides are not equivalent.
// NOLINTNEXTLINE(misc-redundant-expression)
template <typename LHS, typename RHS>
inline constexpr bool binary_vec_op_v = is_vectype_v<LHS> || is_vectype_v<RHS>;

template <typename LHS, typename RHS>
using binary_vec_result_t = std::conditional_t<is_vectype_v<LHS>, LHS, RHS>;

namespace impl {
  // Indexed component accessor
  GCXX_TEMPLATE(int I, typename V)
  GCXX_REQUIRES(is_vectype_v<remove_cvref_t<V>>)
  GCXX_FHDC auto& vec_comp(V& v) {
    if constexpr (I == 0) {
      return v.x;
    } else if constexpr (I == 1) {
      return v.y;
    } else if constexpr (I == 2) {
      return v.z;
    } else {
      static_assert(I == 3, "vector types have at most four components");
      return v.w;
    }
  }

  // Scalar overload
  GCXX_TEMPLATE(int I, typename S)
  GCXX_REQUIRES(!is_vectype_v<S>)
  GCXX_FHDC S vec_comp(S scalar) {
    return scalar;
  }

  template <typename A, typename B, typename Op, std::size_t... Is>
  GCXX_FHDC auto vec_apply(const A& a, const B& b, Op op,
                           std::index_sequence<Is...>) {
    binary_vec_result_t<A, B> result{};
    ((vec_comp<Is>(result) = op(vec_comp<Is>(a), vec_comp<Is>(b))), ...);
    return result;
  }

  // A must be the vector operand (it receives the writes).
  template <typename A, typename B, typename Op, std::size_t... Is>
  GCXX_FHDC A& vec_apply_inplace(A& a, const B& b, Op op,
                                 std::index_sequence<Is...>) {
    ((vec_comp<Is>(a) = op(vec_comp<Is>(a), vec_comp<Is>(b))), ...);
    return a;
  }

  template <typename LHS, typename RHS, typename Op>
  GCXX_FHDC auto apply_binary_dispatch(const LHS& lhs, const RHS& rhs, Op op) {
    constexpr bool lhs_is_vec = is_vectype_v<LHS>;
    constexpr bool rhs_is_vec = is_vectype_v<RHS>;

    static_assert(lhs_is_vec || rhs_is_vec,
                  "vector operators requires at least one vector operand");

    // Two vector operands must have matching component counts
    if constexpr (lhs_is_vec && rhs_is_vec) {
      static_assert(vec_traits<LHS>::size == vec_traits<RHS>::size,
                    "vector operators require matching component counts");
    }

    using traits = vec_traits<binary_vec_result_t<LHS, RHS>>;

    using base_t    = typename traits::value_type;
    constexpr int N = traits::size;

    auto op_base = [&](base_t a, base_t b) {
      return op(a, b);
    };

    // Scalars convert to the base type at the op boundary; vector operands
    // keep their components.
    if constexpr (!lhs_is_vec) {
      static_assert(std::is_convertible_v<LHS, base_t>,
                    "scalar must be convertible to base type");
    }
    if constexpr (!rhs_is_vec) {
      static_assert(std::is_convertible_v<RHS, base_t>,
                    "scalar must be convertible to base type");
    }

    return vec_apply(lhs, rhs, op_base, std::make_index_sequence<N>{});
  }

  template <typename LHS, typename RHS, typename Op>
  GCXX_FHDC LHS& apply_inplace_dispatch(LHS& lhs, const RHS& rhs, Op op) {
    constexpr bool lhs_is_vec = is_vectype_v<LHS>;

    static_assert(lhs_is_vec,
                  "inplace vector operators requires lhs to be vector");

    // A vector rhs must match lhs's component count
    if constexpr (is_vectype_v<RHS>) {
      static_assert(vec_traits<LHS>::size == vec_traits<RHS>::size,
                    "inplace operators require matching component counts");
    }

    using traits = vec_traits<LHS>;

    using base_t = typename traits::value_type;

    constexpr int N = traits::size;

    auto op_base = [&](base_t a, base_t b) {
      return op(a, b);
    };

    // A scalar rhs must be materialized exactly once (e.g. v += v.x) to prevent
    // suprise results
    if constexpr (!is_vectype_v<RHS>) {
      static_assert(std::is_convertible_v<RHS, base_t>,
                    "scalar must be convertible to base type");
      const base_t s_rhs = rhs;
      return vec_apply_inplace(lhs, s_rhs, op_base,
                               std::make_index_sequence<N>{});
    } else {
      return vec_apply_inplace(lhs, rhs, op_base,
                               std::make_index_sequence<N>{});
    }
  }

  namespace op {
    struct product {
      template <typename VT>
      GCXX_FHDC auto operator()(VT a, VT b) -> VT {
        return a * b;
      }
    };

    struct sum {
      template <typename VT>
      GCXX_FHDC auto operator()(VT a, VT b) -> VT {
        return a + b;
      }
    };

    struct difference {
      template <typename VT>
      GCXX_FHDC auto operator()(VT a, VT b) -> VT {
        return a - b;
      }
    };

    struct quotient {
      template <typename VT>
      GCXX_FHDC auto operator()(VT a, VT b) -> VT {
        return a / b;
      }
    };

    struct remainder {
      GCXX_TEMPLATE(typename VT)
      GCXX_REQUIRES(std::is_integral_v<VT>)
      GCXX_FHDC auto operator()(VT a, VT b) -> VT { return a % b; }
    };
  }  // namespace op

}  // namespace impl

GCXX_NAMESPACE_MAIN_DETAILS_END()

GCXX_TEMPLATE(typename LHS, typename RHS)
GCXX_REQUIRES(gcxx::details_::binary_vec_op_v<LHS, RHS>)
GCXX_FHDC auto operator+(const LHS& lhs, const RHS& rhs)
  -> gcxx::details_::binary_vec_result_t<LHS, RHS> {
  using gcxx::details_::impl::apply_binary_dispatch;
  using gcxx::details_::impl::op::sum;
  return apply_binary_dispatch(lhs, rhs, sum{});
}

GCXX_TEMPLATE(typename LHS, typename RHS)
GCXX_REQUIRES(gcxx::details_::binary_vec_op_v<LHS, RHS>)
GCXX_FHDC auto operator-(const LHS& lhs, const RHS& rhs)
  -> gcxx::details_::binary_vec_result_t<LHS, RHS> {
  using gcxx::details_::impl::apply_binary_dispatch;
  using gcxx::details_::impl::op::difference;
  return apply_binary_dispatch(lhs, rhs, difference{});
}

GCXX_TEMPLATE(typename LHS, typename RHS)
GCXX_REQUIRES(gcxx::details_::binary_vec_op_v<LHS, RHS>)
GCXX_FHDC auto operator*(const LHS& lhs, const RHS& rhs)
  -> gcxx::details_::binary_vec_result_t<LHS, RHS> {
  using gcxx::details_::impl::apply_binary_dispatch;
  using gcxx::details_::impl::op::product;
  return apply_binary_dispatch(lhs, rhs, product{});
}

GCXX_TEMPLATE(typename LHS, typename RHS)
GCXX_REQUIRES(gcxx::details_::binary_vec_op_v<LHS, RHS>)
GCXX_FHDC auto operator/(const LHS& lhs, const RHS& rhs)
  -> gcxx::details_::binary_vec_result_t<LHS, RHS> {
  using gcxx::details_::impl::apply_binary_dispatch;
  using gcxx::details_::impl::op::quotient;
  return apply_binary_dispatch(lhs, rhs, quotient{});
}

GCXX_TEMPLATE(typename LHS, typename RHS)
GCXX_REQUIRES(gcxx::details_::binary_vec_op_v<LHS, RHS>)
GCXX_FHDC auto operator%(const LHS& lhs, const RHS& rhs)
  -> gcxx::details_::binary_vec_result_t<LHS, RHS> {
  // Remainder is an integral-only
  using elem_t = typename gcxx::details_::vec_traits<std::conditional_t<
    gcxx::details_::is_vectype_v<LHS>, LHS, RHS>>::value_type;
  static_assert(std::is_integral_v<elem_t>,
                "vector operator% only supports integral element types");
  using gcxx::details_::impl::apply_binary_dispatch;
  using gcxx::details_::impl::op::remainder;
  return apply_binary_dispatch(lhs, rhs, remainder{});
}

GCXX_TEMPLATE(typename LHS, typename RHS)
GCXX_REQUIRES(gcxx::details_::is_vectype_v<LHS>)
GCXX_FHDC auto operator+=(LHS& lhs, const RHS& rhs) -> LHS& {
  using gcxx::details_::impl::apply_inplace_dispatch;
  using gcxx::details_::impl::op::sum;
  return apply_inplace_dispatch(lhs, rhs, sum{});
}

GCXX_TEMPLATE(typename LHS, typename RHS)
GCXX_REQUIRES(gcxx::details_::is_vectype_v<LHS>)
GCXX_FHDC auto operator-=(LHS& lhs, const RHS& rhs) -> LHS& {
  using gcxx::details_::impl::apply_inplace_dispatch;
  using gcxx::details_::impl::op::difference;
  return apply_inplace_dispatch(lhs, rhs, difference{});
}

GCXX_TEMPLATE(typename LHS, typename RHS)
GCXX_REQUIRES(gcxx::details_::is_vectype_v<LHS>)
GCXX_FHDC auto operator/=(LHS& lhs, const RHS& rhs) -> LHS& {

  using gcxx::details_::impl::apply_inplace_dispatch;
  using gcxx::details_::impl::op::quotient;
  return apply_inplace_dispatch(lhs, rhs, quotient{});
}

GCXX_TEMPLATE(typename LHS, typename RHS)
GCXX_REQUIRES(gcxx::details_::is_vectype_v<LHS>)
GCXX_FHDC auto operator*=(LHS& lhs, const RHS& rhs) -> LHS& {
  using gcxx::details_::impl::apply_inplace_dispatch;
  using gcxx::details_::impl::op::product;
  return apply_inplace_dispatch(lhs, rhs, product{});
}

GCXX_TEMPLATE(typename LHS, typename RHS)
GCXX_REQUIRES(gcxx::details_::is_vectype_v<LHS>)
GCXX_FHDC auto operator%=(LHS& lhs, const RHS& rhs) -> LHS& {
  // Remainder is an integral-only
  using elem_t = typename gcxx::details_::vec_traits<LHS>::value_type;
  static_assert(std::is_integral_v<elem_t>,
                "vector operator%= only supports integral element types");
  using gcxx::details_::impl::apply_inplace_dispatch;
  using gcxx::details_::impl::op::remainder;
  return apply_inplace_dispatch(lhs, rhs, remainder{});
}

// TODO: Use expression templates to avoid temporaries (CppCon 2019 talk).


#endif