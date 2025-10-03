#include <type_traits>

#include <gtest/gtest.h>
#include <gcxx/runtime/event.hpp>

// Test fixture for event_base tests
class EventBaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup code if needed
  }

  void TearDown() override {
    // Cleanup code if needed
  }
};

// Test default construction
TEST_F(EventBaseTest, DefaultConstructor) {
  gcxx::details_::event_wrap event;
  
  // Default constructed event should be invalid
  EXPECT_FALSE(static_cast<bool>(event));
  EXPECT_EQ(event.get(), gcxx::details_::INVALID_EVENT);
}

// Test construction from raw device event
TEST_F(EventBaseTest, RawEventConstructor) {
  // Create a mock device event (assuming cudaEvent_t is a pointer type)
  gcxx::details_::deviceEvent_t raw_event = reinterpret_cast<gcxx::details_::deviceEvent_t>(0x12345);
  
  gcxx::details_::event_wrap event(raw_event);
  
  EXPECT_TRUE(static_cast<bool>(event));
  EXPECT_EQ(event.get(), raw_event);
}

// Test that construction from int is deleted
TEST_F(EventBaseTest, IntConstructorDeleted) {
  // This test verifies that the deleted constructor is indeed deleted
  // We can't instantiate it, but we can check with type traits
  EXPECT_FALSE((std::is_constructible_v<gcxx::details_::event_wrap, int>));
}

// Test that construction from nullptr is deleted
TEST_F(EventBaseTest, NullptrConstructorDeleted) {
  // This test verifies that the deleted constructor is indeed deleted
  EXPECT_FALSE((std::is_constructible_v<gcxx::details_::event_wrap, std::nullptr_t>));
}

// Test get() method
TEST_F(EventBaseTest, GetMethod) {
  gcxx::details_::deviceEvent_t raw_event = reinterpret_cast<gcxx::details_::deviceEvent_t>(0xABCDE);
  gcxx::details_::event_wrap event(raw_event);
  
  EXPECT_EQ(event.get(), raw_event);
}

// Test implicit conversion to deviceEvent_t
TEST_F(EventBaseTest, ImplicitConversion) {
  gcxx::details_::deviceEvent_t raw_event = reinterpret_cast<gcxx::details_::deviceEvent_t>(0xFEDCB);
  gcxx::details_::event_wrap event(raw_event);
  
  gcxx::details_::deviceEvent_t converted = event;
  EXPECT_EQ(converted, raw_event);
}

// Test explicit bool conversion - valid event
TEST_F(EventBaseTest, BoolConversionValid) {
  gcxx::details_::deviceEvent_t raw_event = reinterpret_cast<gcxx::details_::deviceEvent_t>(0x54321);
  gcxx::details_::event_wrap event(raw_event);
  
  EXPECT_TRUE(static_cast<bool>(event));
  EXPECT_TRUE(event);
}

// Test explicit bool conversion - invalid event
TEST_F(EventBaseTest, BoolConversionInvalid) {
  gcxx::details_::event_wrap event;  // Default constructed (invalid)
  
  EXPECT_FALSE(static_cast<bool>(event));
  EXPECT_FALSE(event);
}

// Test equality operator
TEST_F(EventBaseTest, EqualityOperator) {
  gcxx::details_::deviceEvent_t raw_event1 = reinterpret_cast<gcxx::details_::deviceEvent_t>(0x11111);
  gcxx::details_::deviceEvent_t raw_event2 = reinterpret_cast<gcxx::details_::deviceEvent_t>(0x22222);
  
  gcxx::details_::event_wrap event1(raw_event1);
  gcxx::details_::event_wrap event2(raw_event1);  // Same raw event
  gcxx::details_::event_wrap event3(raw_event2);  // Different raw event
  
  EXPECT_TRUE(event1 == event2);   // Same underlying event
  EXPECT_FALSE(event1 == event3);  // Different underlying events
}

// Test inequality operator
TEST_F(EventBaseTest, InequalityOperator) {
  gcxx::details_::deviceEvent_t raw_event1 = reinterpret_cast<gcxx::details_::deviceEvent_t>(0x33333);
  gcxx::details_::deviceEvent_t raw_event2 = reinterpret_cast<gcxx::details_::deviceEvent_t>(0x44444);
  
  gcxx::details_::event_wrap event1(raw_event1);
  gcxx::details_::event_wrap event2(raw_event1);  // Same raw event
  gcxx::details_::event_wrap event3(raw_event2);  // Different raw event
  
  EXPECT_FALSE(event1 != event2);  // Same underlying event
  EXPECT_TRUE(event1 != event3);   // Different underlying events
}

// Test equality with invalid events
TEST_F(EventBaseTest, EqualityWithInvalidEvents) {
  gcxx::details_::event_wrap invalid1;  // Default constructed
  gcxx::details_::event_wrap invalid2;  // Default constructed
  gcxx::details_::deviceEvent_t raw_event = reinterpret_cast<gcxx::details_::deviceEvent_t>(0x55555);
  gcxx::details_::event_wrap valid(raw_event);
  
  EXPECT_TRUE(invalid1 == invalid2);   // Both invalid should be equal
  EXPECT_FALSE(invalid1 == valid);     // Invalid != valid
  EXPECT_TRUE(invalid1 != valid);      // Invalid != valid
}

// Test copy constructor
TEST_F(EventBaseTest, CopyConstructor) {
  gcxx::details_::deviceEvent_t raw_event = reinterpret_cast<gcxx::details_::deviceEvent_t>(0x66666);
  gcxx::details_::event_wrap original(raw_event);
  gcxx::details_::event_wrap copy(original);
  
  EXPECT_EQ(original.get(), copy.get());
  EXPECT_TRUE(original == copy);
}

// Test copy assignment
TEST_F(EventBaseTest, CopyAssignment) {
  gcxx::details_::deviceEvent_t raw_event = reinterpret_cast<gcxx::details_::deviceEvent_t>(0x77777);
  gcxx::details_::event_wrap original(raw_event);
  gcxx::details_::event_wrap assigned;
  
  assigned = original;
  
  EXPECT_EQ(original.get(), assigned.get());
  EXPECT_TRUE(original == assigned);
}

// Test constexpr functionality
TEST_F(EventBaseTest, ConstexprFunctionality) {
  // Test that INVALID_EVENT is constexpr
  constexpr auto invalid = gcxx::details_::INVALID_EVENT;
  (void)invalid;  // Suppress unused variable warning

  // Test constexpr constructor
  constexpr gcxx::details_::event_wrap default_event;
  EXPECT_FALSE(static_cast<bool>(default_event));
}

// Test type traits
TEST_F(EventBaseTest, TypeTraits) {
  using event_type = gcxx::details_::event_wrap;
  
  // Check basic type traits
  EXPECT_TRUE(std::is_default_constructible_v<event_type>);
  EXPECT_TRUE(std::is_copy_constructible_v<event_type>);
  EXPECT_TRUE(std::is_copy_assignable_v<event_type>);
  EXPECT_TRUE(std::is_move_constructible_v<event_type>);
  EXPECT_TRUE(std::is_move_assignable_v<event_type>);
  
  // Check that int constructor is deleted
  EXPECT_FALSE((std::is_constructible_v<event_type, int>));
  
  // Check that nullptr_t constructor is deleted
  EXPECT_FALSE((std::is_constructible_v<event_type, std::nullptr_t>));
  
  // Check that raw event constructor exists
  EXPECT_TRUE((std::is_constructible_v<event_type, gcxx::details_::deviceEvent_t>));
}

