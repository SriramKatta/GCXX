// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
// Exercises the GCXX concept DSL ported from CCCL's concept_macros.h; built
// twice (C++17 SFINAE and C++20 concepts) so both branches are covered.
#include "tests_common.hpp"

#include <memory>

#include <gcxx/macros/template_helper_macros.hpp>
#include <gcxx/runtime/details/type_traits.hpp>

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

  // GCXX_REQUIRES_EXPR: inline requires-body as a concept's last requirement.
  template <class T>
  GCXX_CONCEPT has_eq_neq =
    GCXX_REQUIRES_EXPR((T), T const& a, T const& b)(a == b, a != b);

  // Fragment exercising every requirement form (_Same_as targets prvalue).
  template <class T, class U>
  GCXX_CONCEPT_FRAGMENT(
    frag_, requires(T t, U u)(requires(has_eq_neq<T>), requires(has_eq_neq<U>),
                              typename(T), noexcept(t == u), _Same_as(T)(T{}),
                              _Satisfies(has_eq_neq) t, (t == u), t != u));
  template <class T, class U>
  GCXX_CONCEPT frag = GCXX_FRAGMENT(frag_, T, U);

}  // namespace

// Compile-time positive/negative checks covering both macro branches.
static_assert(has_eq_neq<int>, "");
static_assert(has_eq_neq<Comparable>, "");
static_assert(!has_eq_neq<Incomparable>, "");
static_assert(frag<Comparable, Comparable>, "");
static_assert(!frag<Incomparable, Comparable>, "");
static_assert(!frag<Comparable, Incomparable>, "");

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

// Regression: GCXX_TEMPLATE/GCXX_REQUIRES still constrain ~30 call sites.
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
