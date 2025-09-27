#include <type_traits>

#include <gtest/gtest.h>
#include <gpucxx/runtime/event/event_base.hpp>

using namespace gcxx::details_;  //since no other file includes this

// Parameterized fixture
class EventRefParameterizedTest
    : public ::testing::TestWithParam<deviceEvent_t> {};

INSTANTIATE_TEST_SUITE_P(
  EventHandles, EventRefParameterizedTest,
  ::testing::Values(reinterpret_cast<deviceEvent_t>(0x0),     // null-like
                    reinterpret_cast<deviceEvent_t>(0x1),     // minimal valid
                    reinterpret_cast<deviceEvent_t>(0x1234),  // arbitrary value
                    reinterpret_cast<deviceEvent_t>(0xFFFF)   // high value
                    ));

TEST_P(EventRefParameterizedTest, ConstructAndGet) {
  deviceEvent_t raw = GetParam();
  event_ref e(raw);

  EXPECT_EQ(e.get(), raw);
  if (raw == INVALID_EVENT) {
    EXPECT_FALSE(static_cast<bool>(e));
  } else {
    EXPECT_TRUE(static_cast<bool>(e));
  }
}

TEST_P(EventRefParameterizedTest, ConversionOperator) {
  deviceEvent_t raw = GetParam();
  event_ref e(raw);

  deviceEvent_t converted = static_cast<deviceEvent_t>(e);
  EXPECT_EQ(converted, raw);
}

TEST_P(EventRefParameterizedTest, EqualityAndInequality) {
  deviceEvent_t raw = GetParam();
  event_ref e1(raw);
  event_ref e2(raw);

  EXPECT_TRUE(e1 == e2);
  EXPECT_FALSE(e1 != e2);

  // Compare against different event
  deviceEvent_t other = reinterpret_cast<deviceEvent_t>(0xBEEF);
  if (raw != other) {
    event_ref e3(other);
    EXPECT_TRUE(e1 != e3);
  }
}

// Non-parameterized test for default constructor
TEST(EventRefStandaloneTest, DefaultConstructorCreatesInvalidEvent) {
  event_ref e;
  EXPECT_FALSE(static_cast<bool>(e));
  EXPECT_EQ(e.get(), INVALID_EVENT);
}

TEST(EventRefStandaloneTest, DeletedConstructors) {
  static_assert(!std::is_constructible_v<event_ref, int>);
  static_assert(!std::is_constructible_v<event_ref, std::nullptr_t>);
  static_assert(std::is_copy_constructible_v<event_ref>);
  static_assert(std::is_copy_assignable_v<event_ref>);
  SUCCEED();  // runtime pass
}
