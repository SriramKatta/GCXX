// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Exercises the GCXX concept DSL ported from CCCL's concept_macros.h:
// GCXX_REQUIRES_EXPR, GCXX_CONCEPT_FRAGMENT / GCXX_FRAGMENT, and every
// requirement form (requires / typename / noexcept / _Same_as / _Satisfies /
// (EXPR) / EXPR).
//
// The same source is built twice (see CMakeLists.txt):
//   * CUDA_STANDARD 17 -> pre-C++20 SFINAE-emulation branch
//   (!GCXX_HAS_CONCEPTS)
//   * CUDA_STANDARD 20 -> C++20 requires(...) branch        (
//   GCXX_HAS_CONCEPTS)
// so both implementations of every macro are covered.
#include "tests_common.hpp"

#include <memory>

#include <gcxx/macros/template_helper_macros.hpp>
#include <gcxx/runtime/details/type_traits.hpp>

// TODO : need to check this even for hipcc
// #ifdef __CUDACC__
// #  pragma nv_diag_suppress 177
// #endif

namespace {

  struct Comparable  // has == / !=, both noexcept
  {
    int v{};
    [[maybe_unused]] friend bool operator==(Comparable, Comparable) noexcept {
      return true;
    }
    [[maybe_unused]] friend bool operator!=(Comparable, Comparable) noexcept {
      return false;
    }
  };
  struct Incomparable {};  // no relational operators

  // (1) GCXX_REQUIRES_EXPR: inline requires-body, the last requirement of a
  // concept.
  template <class T>
  GCXX_CONCEPT has_eq_neq =
    GCXX_REQUIRES_EXPR((T), T const& a, T const& b)(a == b, a != b);

  // (2) GCXX_CONCEPT_FRAGMENT + GCXX_FRAGMENT: a named fragment exercising
  // every requirement form.
  //
  // _Same_as targets a prvalue (T{}) so the requirement holds in BOTH the C++17
  // (decltype(EXPR)) and C++20 (decltype((EXPR))) branches; an lvalue would
  // differ because the C++20 compound-requirement rule yields a reference type.
  // This reference-vs-value behavior matches CCCL exactly.
  template <class T, class U>
  GCXX_CONCEPT_FRAGMENT(
    frag_, requires(T t, U u)(requires(has_eq_neq<T>), requires(has_eq_neq<U>),
                              typename(T), noexcept(t == u), _Same_as(T)(T{}),
                              _Satisfies(has_eq_neq) t, (t == u), t != u));
  template <class T, class U>
  GCXX_CONCEPT frag = GCXX_FRAGMENT(frag_, T, U);

}  // namespace

// Positive / negative checks at compile time (these are the real test of both
// branches: the static_asserts must hold whether concepts are emulated or
// native).
static_assert(has_eq_neq<int>, "");
static_assert(has_eq_neq<Comparable>, "");
static_assert(!has_eq_neq<Incomparable>, "");
static_assert(frag<Comparable, Comparable>, "");
static_assert(!frag<Incomparable, Comparable>, "");
static_assert(!frag<Comparable, Incomparable>, "");

// The gcxx::details_:: detector traits are now built on the concept DSL.
// Forward-declared only -> not complete.
struct ForwardDeclaredOnly;
static_assert(gcxx::details_::is_complete_v<int>, "");
static_assert(!gcxx::details_::is_complete_v<ForwardDeclaredOnly>, "");
static_assert(gcxx::details_::has_get_to_pointer_v<std::unique_ptr<int>>, "");
static_assert(!gcxx::details_::has_get_to_pointer_v<int>, "");
static_assert(gcxx::details_::is_pointer_or_has_get_v<int*>, "");
static_assert(gcxx::details_::is_pointer_or_has_get_v<std::unique_ptr<int>>,
              "");
static_assert(!gcxx::details_::is_pointer_or_has_get_v<int>, "");

TEST(GcxxConceptMacros, RequiresExpr) {
  EXPECT_TRUE(has_eq_neq<int>);
  EXPECT_TRUE(has_eq_neq<Comparable>);
  EXPECT_FALSE(has_eq_neq<Incomparable>);
}

TEST(GcxxConceptMacros, Fragment) {
  EXPECT_TRUE((frag<Comparable, Comparable>));
  EXPECT_FALSE((frag<Incomparable, Comparable>));
  EXPECT_FALSE((frag<Comparable, Incomparable>));
}

// (3) Regression: the pre-existing GCXX_TEMPLATE / GCXX_REQUIRES still
// constrain overloads (used by ~30 existing call sites in the library).
GCXX_TEMPLATE(typename T)
GCXX_REQUIRES(has_eq_neq<T>)
T echo(T x) {
  return x;
}

TEST(GcxxConceptMacros, TemplateRequires) {
  EXPECT_EQ(echo(Comparable{5}).v, 5);
  EXPECT_EQ(echo(7), 7);
}

// Detector traits rewritten on top of the concept DSL.
TEST(GcxxConceptMacros, ConvertedTypeTraits) {
  EXPECT_TRUE(gcxx::details_::is_complete_v<int>);
  EXPECT_TRUE(gcxx::details_::has_get_to_pointer_v<std::unique_ptr<int>>);
  EXPECT_TRUE(gcxx::details_::is_pointer_or_has_get_v<int*>);
  EXPECT_TRUE(gcxx::details_::is_pointer_or_has_get_v<std::unique_ptr<int>>);
  EXPECT_FALSE(gcxx::details_::has_get_to_pointer_v<int>);
  EXPECT_FALSE(gcxx::details_::is_pointer_or_has_get_v<int>);
}
