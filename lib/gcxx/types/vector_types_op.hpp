#pragma once
#ifndef GCXX_TYPES_VECTOR_TYPES_OP_HPP
#define GCXX_TYPES_VECTOR_TYPES_OP_HPP

#include <cmath>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/types/vector_types.hpp>
#include <type_traits>


GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

namespace impl {
  template <int N>
  struct vec_op_impl {
    //  for vector ⊗ vector  where scalar is on the left
    template <typename V, typename Op>
    GCXX_FHDC static V apply_componentwise(const V& a, const V& b, Op op) {
      V result{};
      result.x = op(a.x, b.x);
      if constexpr (N >= 2)
        result.y = op(a.y, b.y);
      if constexpr (N >= 3)
        result.z = op(a.z, b.z);
      if constexpr (N >= 4)
        result.w = op(a.w, b.w);
      return result;
    }

    //  for vector ⊗ scalar  where scalar is on the left
    template <typename V, typename S, typename Op>
    GCXX_FHDC static V apply_scalar(const V& v, S scalar, Op op) {
      V result{};
      result.x = op(v.x, scalar);
      if constexpr (N >= 2)
        result.y = op(v.y, scalar);
      if constexpr (N >= 3)
        result.z = op(v.z, scalar);
      if constexpr (N >= 4)
        result.w = op(v.w, scalar);
      return result;
    }

    //  for scalar ⊗ vector  where scalar is on the left
    template <typename V, typename S, typename Op>
    GCXX_FHDC static V apply_scalar_left(S scalar, const V& v, Op op) {
      V result{};
      result.x = op(scalar, v.x);
      if constexpr (N >= 2)
        result.y = op(scalar, v.y);
      if constexpr (N >= 3)
        result.z = op(scalar, v.z);
      if constexpr (N >= 4)
        result.w = op(scalar, v.w);
      return result;
    }

    template <typename V, typename Op>
    GCXX_FHDC static V& apply_inplace_componentwise(V& a, const V& b, Op op) {
      a.x = op(a.x, b.x);
      if constexpr (N >= 2)
        a.y = op(a.y, b.y);
      if constexpr (N >= 3)
        a.z = op(a.z, b.z);
      if constexpr (N >= 4)
        a.w = op(a.w, b.w);
      return a;
    }

    //  for vector ⊗ scalar  where scalar is on the left
    template <typename V, typename S, typename Op>
    GCXX_FHDC static V& apply_inplace_scalar(V& v, const S scalar, Op op) {
      v.x = op(v.x, scalar);
      if constexpr (N >= 2)
        v.y = op(v.y, scalar);
      if constexpr (N >= 3)
        v.z = op(v.z, scalar);
      if constexpr (N >= 4)
        v.w = op(v.w, scalar);
      return v;
    }
  };

  template <typename LHS, typename RHS, typename Op>
  GCXX_FHDC auto apply_binary_dispatch(const LHS& lhs, const RHS& rhs, Op op) {
    constexpr bool lhs_is_vec = is_vectype_v<LHS>;
    constexpr bool rhs_is_vec = is_vectype_v<RHS>;

    GCXX_STATIC_EXPECT(lhs_is_vec || rhs_is_vec,
                       "vector operators requires at least one vector operand");

    using traits =
      std::conditional_t<lhs_is_vec, vec_traits<LHS>, vec_traits<RHS>>;

    using base_t    = typename traits::value_type;
    constexpr int N = traits::size;

    auto op_base = [&](base_t a, base_t b) {
      return op(a, b);
    };

    if constexpr (lhs_is_vec && rhs_is_vec) {
      return vec_op_impl<N>::apply_componentwise(lhs, rhs, op_base);
    } else if constexpr (lhs_is_vec) {
      static_assert(std::is_convertible_v<RHS, base_t>,
                    "scalar must be convertible to base type");
      return vec_op_impl<N>::apply_scalar(lhs, rhs, op_base);
    } else if constexpr (rhs_is_vec) {
      static_assert(std::is_convertible_v<LHS, base_t>,
                    "scalar must be convertible to base type");
      return vec_op_impl<N>::apply_scalar_left(lhs, rhs, op_base);
    }
  }

  template <typename LHS, typename RHS, typename Op>
  GCXX_FHDC LHS& apply_inplace_dispatch(LHS& lhs, const RHS& rhs, Op op) {
    constexpr bool lhs_is_vec = is_vectype_v<LHS>;
    constexpr bool rhs_is_vec = is_vectype_v<RHS>;

    GCXX_STATIC_EXPECT(lhs_is_vec,
                       "inplace vector operators requires lhs to be vector");

    using traits = vec_traits<LHS>;

    using base_t = typename traits::value_type;

    constexpr int N = traits::size;

    auto op_base = [&](base_t a, base_t b) {
      return op(a, b);
    };

    if constexpr (rhs_is_vec) {
      return vec_op_impl<N>::apply_inplace_componentwise(lhs, rhs, op_base);
    } else {
      static_assert(std::is_convertible_v<RHS, base_t>,
                    "scalar must be convertible to base type");
      return vec_op_impl<N>::apply_inplace_scalar(lhs, rhs, op_base);
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
      template <typename VT>
      GCXX_FHDC auto operator()(VT a, VT b)
        -> std::enable_if_t<std::is_integral_v<VT>, VT> {
        return a % b;
      }
    };
  }  // namespace op

}  // namespace impl

template <typename LHS, typename RHS>
inline constexpr bool binary_vec_op_v = is_vectype_v<LHS> || is_vectype_v<RHS>;

template <typename LHS, typename RHS>
using binary_vec_result_t = std::conditional_t<is_vectype_v<LHS>, LHS, RHS>;


GCXX_NAMESPACE_MAIN_DETAILS_END

/**
 * @brief
 * unary plus over load the vecdata type is always maintained
 * @tparam LHS
 * @tparam RHS
 * @param lhs
 * @param rhs
 * @return gcxx::details_::binary_vec_result_t<LHS, RHS>
 */
template <typename LHS, typename RHS,
          std::enable_if_t<gcxx::details_::binary_vec_op_v<LHS, RHS>, int> = 0>
GCXX_FHDC auto operator+(const LHS& lhs, const RHS& rhs)
  -> gcxx::details_::binary_vec_result_t<LHS, RHS> {
  using namespace gcxx::details_::impl;
  return apply_binary_dispatch(lhs, rhs, op::sum{});
}

template <typename LHS, typename RHS,
          std::enable_if_t<gcxx::details_::binary_vec_op_v<LHS, RHS>, int> = 0>
GCXX_FHDC auto operator-(const LHS& lhs, const RHS& rhs)
  -> gcxx::details_::binary_vec_result_t<LHS, RHS> {
  using namespace gcxx::details_::impl;
  return apply_binary_dispatch(lhs, rhs, op::difference{});
}

template <typename LHS, typename RHS,
          std::enable_if_t<gcxx::details_::binary_vec_op_v<LHS, RHS>, int> = 0>
GCXX_FHDC auto operator*(const LHS& lhs, const RHS& rhs)
  -> gcxx::details_::binary_vec_result_t<LHS, RHS> {
  using namespace gcxx::details_::impl;
  return apply_binary_dispatch(lhs, rhs, op::product{});
}

template <typename LHS, typename RHS,
          std::enable_if_t<gcxx::details_::binary_vec_op_v<LHS, RHS>, int> = 0>
GCXX_FHDC auto operator/(const LHS& lhs, const RHS& rhs)
  -> gcxx::details_::binary_vec_result_t<LHS, RHS> {
  using namespace gcxx::details_::impl;
  return apply_binary_dispatch(lhs, rhs, op::quotient{});
}

template <typename LHS, typename RHS,
          std::enable_if_t<gcxx::details_::binary_vec_op_v<LHS, RHS>, int> = 0>
GCXX_FHDC auto operator%(const LHS& lhs, const RHS& rhs)
  -> gcxx::details_::binary_vec_result_t<LHS, RHS> {
  using namespace gcxx::details_::impl;
  return apply_binary_dispatch(lhs, rhs, op::remainder{});
}

template <typename LHS, typename RHS,
          std::enable_if_t<gcxx::details_::is_vectype_v<LHS>, int> = 0>
GCXX_FHDC auto operator+=(LHS& lhs, const RHS& rhs) -> LHS& {
  using namespace gcxx::details_::impl;
  return apply_inplace_dispatch(lhs, rhs, op::sum{});
}

template <typename LHS, typename RHS,
          std::enable_if_t<gcxx::details_::is_vectype_v<LHS>, int> = 0>
GCXX_FHDC auto operator-=(LHS& lhs, const RHS& rhs) -> LHS& {
  using namespace gcxx::details_::impl;
  return apply_inplace_dispatch(lhs, rhs, op::difference{});
}

template <typename LHS, typename RHS,
          std::enable_if_t<gcxx::details_::is_vectype_v<LHS>, int> = 0>
GCXX_FHDC auto operator/=(LHS& lhs, const RHS& rhs) -> LHS& {

  using namespace gcxx::details_::impl;
  return apply_inplace_dispatch(lhs, rhs, op::quotient{});
}

template <typename LHS, typename RHS,
          std::enable_if_t<gcxx::details_::is_vectype_v<LHS>, int> = 0>
GCXX_FHDC auto operator*=(LHS& lhs, const RHS& rhs) -> LHS& {
  using namespace gcxx::details_::impl;
  return apply_inplace_dispatch(lhs, rhs, op::product{});
}

template <typename LHS, typename RHS,
          std::enable_if_t<gcxx::details_::is_vectype_v<LHS>, int> = 0>
GCXX_FHDC auto operator%=(LHS& lhs, const RHS& rhs) -> LHS& {
  using namespace gcxx::details_::impl;
  return apply_inplace_dispatch(lhs, rhs, op::remainder{});
}


#endif