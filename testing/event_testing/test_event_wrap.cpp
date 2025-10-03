#include <gtest/gtest.h>
#include <chrono>
#include <gcxx/runtime/event.hpp>
#include <type_traits>

using namespace gcxx;

class EventWrapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    GCXX_SAFE_RUNTIME_CALL(StreamCreate, (&stream_));
    GCXX_SAFE_RUNTIME_CALL(EventCreate, (&raw_event_));
  }

  void TearDown() override {
    GCXX_SAFE_RUNTIME_CALL(EventDestroy, (raw_event_));
    GCXX_SAFE_RUNTIME_CALL(StreamDestroy, (stream_));
  }

  GCXX_RUNTIME_BACKEND(Stream_t) stream_{};
  GCXX_RUNTIME_BACKEND(Event_t) raw_event_{};
};

// Test default construction
TEST_F(EventWrapTest, DefaultConstructor) {
  event_wrap event;

  // Default constructed event should be invalid
  EXPECT_FALSE(static_cast<bool>(event));
  EXPECT_EQ(event.get(), details_::INVALID_EVENT);
}

// Test construction from raw device event
TEST_F(EventWrapTest, RawEventConstructor) {
  event_wrap event(raw_event_);

  EXPECT_TRUE(static_cast<bool>(event));
  EXPECT_EQ(event.get(), raw_event_);
}

// Test construction from event_base
TEST_F(EventWrapTest, EventBaseConstructor) {
  details_::event_wrap base_event(raw_event_);
  event_wrap event(base_event);

  EXPECT_TRUE(static_cast<bool>(event));
  EXPECT_EQ(event.get(), raw_event_);
  EXPECT_EQ(event.get(), base_event.get());
}

// Test that construction from int is deleted
TEST_F(EventWrapTest, IntConstructorDeleted) {
  EXPECT_FALSE((std::is_constructible_v<event_wrap, int>));
}

// Test that construction from nullptr is deleted
TEST_F(EventWrapTest, NullptrConstructorDeleted) {
  EXPECT_FALSE((std::is_constructible_v<event_wrap, std::nullptr_t>));
}

// Test inheritance from event_base
TEST_F(EventWrapTest, InheritsFromEventBase) {
  EXPECT_TRUE((std::is_base_of_v<details_::event_wrap, event_wrap>));
}

// Test get() method (inherited)
TEST_F(EventWrapTest, GetMethod) {
  event_wrap event(raw_event_);
  EXPECT_EQ(event.get(), raw_event_);
}

// Test implicit conversion to deviceEvent_t (inherited)
TEST_F(EventWrapTest, ImplicitConversion) {
  event_wrap event(raw_event_);
  details_::deviceEvent_t converted = event;
  EXPECT_EQ(converted, raw_event_);
}

// Test explicit bool conversion (inherited)
TEST_F(EventWrapTest, BoolConversion) {
  event_wrap valid_event(raw_event_);
  event_wrap invalid_event;

  EXPECT_TRUE(static_cast<bool>(valid_event));
  EXPECT_FALSE(static_cast<bool>(invalid_event));
}

// Test equality operators (inherited)
TEST_F(EventWrapTest, EqualityOperators) {
  event_wrap event1(raw_event_);
  event_wrap event2(raw_event_);
  event_wrap invalid_event;

  EXPECT_TRUE(event1 == event2);
  EXPECT_FALSE(event1 == invalid_event);
  EXPECT_FALSE(event1 != event2);
  EXPECT_TRUE(event1 != invalid_event);
}

// Test RecordInStream with default stream
TEST_F(EventWrapTest, RecordInStreamDefault) {
  event_wrap event(raw_event_);

  // Should not throw - recording in default stream
  EXPECT_NO_THROW(event.RecordInStream());
}

// Test RecordInStream with specific stream
TEST_F(EventWrapTest, RecordInStreamSpecific) {
  event_wrap event(raw_event_);
  stream_wrap test_stream(stream_);

  // Should not throw - recording in specific stream
  EXPECT_NO_THROW(event.RecordInStream(test_stream));
}

// Test RecordInStream with flags
TEST_F(EventWrapTest, RecordInStreamWithFlags) {
  event_wrap event(raw_event_);
  stream_wrap test_stream(stream_);

  // Test with different record flags
  EXPECT_NO_THROW(event.RecordInStream(test_stream, flags::eventRecord::none));
  // TODO : need to understand why this fails wiith cuda error
  // EXPECT_NO_THROW(event.RecordInStream(test_stream,
  // flags::eventRecord::external));
}

// Test Synchronize
TEST_F(EventWrapTest, Synchronize) {
  event_wrap event(raw_event_);

  // Record the event first
  event.RecordInStream();

  // Should not throw
  EXPECT_NO_THROW(event.Synchronize());
}

// Test HasOccurred
TEST_F(EventWrapTest, HasOccurred) {
  event_wrap event(raw_event_);

  // Record the event
  event.RecordInStream();

  // Event should have occurred after synchronization
  event.Synchronize();
  EXPECT_TRUE(event.HasOccurred());
}

// Test ElapsedTimeSince with milliSec (default)
TEST_F(EventWrapTest, ElapsedTimeSinceMilliSec) {
  GCXX_RUNTIME_BACKEND(Event_t) start_raw, end_raw;
  GCXX_SAFE_RUNTIME_CALL(EventCreate, (&start_raw));
  GCXX_SAFE_RUNTIME_CALL(EventCreate, (&end_raw));

  event_wrap start_event(start_raw);
  event_wrap end_event(end_raw);
  stream_wrap test_stream(stream_);

  // Record events in sequence
  start_event.RecordInStream(test_stream);
  end_event.RecordInStream(test_stream);

  // Get elapsed time
  auto elapsed = end_event.ElapsedTimeSince(start_event);
  EXPECT_GE(elapsed.count(), 0.0f);

  // Check return type
  EXPECT_TRUE((std::is_same_v<decltype(elapsed), milliSec>));

  // Cleanup
  GCXX_SAFE_RUNTIME_CALL(EventDestroy, (start_raw));
  GCXX_SAFE_RUNTIME_CALL(EventDestroy, (end_raw));
}

// Test ElapsedTimeSince with different duration types
TEST_F(EventWrapTest, ElapsedTimeSinceDifferentDurations) {
  GCXX_RUNTIME_BACKEND(Event_t) start_raw, end_raw;
  GCXX_SAFE_RUNTIME_CALL(EventCreate, (&start_raw));
  GCXX_SAFE_RUNTIME_CALL(EventCreate, (&end_raw));

  event_wrap start_event(start_raw);
  event_wrap end_event(end_raw);
  stream_wrap test_stream(stream_);

  // Record events
  start_event.RecordInStream(test_stream);
  end_event.RecordInStream(test_stream);

  // Test different duration types
  auto nano_elapsed  = end_event.ElapsedTimeSince<nanoSec>(start_event);
  auto micro_elapsed = end_event.ElapsedTimeSince<microSec>(start_event);
  auto milli_elapsed = end_event.ElapsedTimeSince<milliSec>(start_event);
  auto sec_elapsed   = end_event.ElapsedTimeSince<sec>(start_event);

  EXPECT_GE(nano_elapsed.count(), 0.0f);
  EXPECT_GE(micro_elapsed.count(), 0.0f);
  EXPECT_GE(milli_elapsed.count(), 0.0f);
  EXPECT_GE(sec_elapsed.count(), 0.0f);

  // Check return types
  EXPECT_TRUE((std::is_same_v<decltype(nano_elapsed), nanoSec>));
  EXPECT_TRUE((std::is_same_v<decltype(micro_elapsed), microSec>));
  EXPECT_TRUE((std::is_same_v<decltype(milli_elapsed), milliSec>));
  EXPECT_TRUE((std::is_same_v<decltype(sec_elapsed), sec>));

  // Cleanup
  GCXX_SAFE_RUNTIME_CALL(EventDestroy, (start_raw));
  GCXX_SAFE_RUNTIME_CALL(EventDestroy, (end_raw));
}

// Test ElapsedTimeBetween static method
TEST_F(EventWrapTest, ElapsedTimeBetweenStatic) {
  GCXX_RUNTIME_BACKEND(Event_t) start_raw, end_raw;
  GCXX_SAFE_RUNTIME_CALL(EventCreate, (&start_raw));
  GCXX_SAFE_RUNTIME_CALL(EventCreate, (&end_raw));

  event_wrap start_event(start_raw);
  event_wrap end_event(end_raw);
  stream_wrap test_stream(stream_);

  // Record events
  start_event.RecordInStream(test_stream);
  end_event.RecordInStream(test_stream);

  // Test static method
  auto elapsed = event_wrap::ElapsedTimeBetween(start_event, end_event);
  EXPECT_GE(elapsed.count(), 0.0f);

  // Test with different duration type
  auto elapsed_nano =
    event_wrap::ElapsedTimeBetween<nanoSec>(start_event, end_event);
  EXPECT_GE(elapsed_nano.count(), 0.0f);
  EXPECT_TRUE((std::is_same_v<decltype(elapsed_nano), nanoSec>));

  // Cleanup
  GCXX_SAFE_RUNTIME_CALL(EventDestroy, (start_raw));
  GCXX_SAFE_RUNTIME_CALL(EventDestroy, (end_raw));
}

// Test duration conversion function
TEST_F(EventWrapTest, DurationConversion) {
  float test_ms = 123.456f;

  auto nano_duration  = ConvertDuration<nanoSec>(test_ms);
  auto micro_duration = ConvertDuration<microSec>(test_ms);
  auto milli_duration = ConvertDuration<milliSec>(test_ms);
  auto sec_duration   = ConvertDuration<sec>(test_ms);

  EXPECT_FLOAT_EQ(nano_duration.count(), test_ms * 1000000.0f);
  EXPECT_FLOAT_EQ(micro_duration.count(), test_ms * 1000.0f);
  EXPECT_FLOAT_EQ(milli_duration.count(), test_ms);
  EXPECT_FLOAT_EQ(sec_duration.count(), test_ms / 1000.0f);
}

// Test copy constructor
TEST_F(EventWrapTest, CopyConstructor) {
  event_wrap original(raw_event_);
  event_wrap copy(original);

  EXPECT_EQ(original.get(), copy.get());
  EXPECT_TRUE(original == copy);
}

// Test copy assignment
TEST_F(EventWrapTest, CopyAssignment) {
  event_wrap original(raw_event_);
  event_wrap assigned;

  assigned = original;

  EXPECT_EQ(original.get(), assigned.get());
  EXPECT_TRUE(original == assigned);
}

// Test type traits
TEST_F(EventWrapTest, TypeTraits) {
  using event_type = event_wrap;

  // Check basic type traits
  EXPECT_TRUE(std::is_default_constructible_v<event_type>);
  EXPECT_TRUE(std::is_copy_constructible_v<event_type>);
  EXPECT_TRUE(std::is_copy_assignable_v<event_type>);
  EXPECT_TRUE(std::is_move_constructible_v<event_type>);
  EXPECT_TRUE(std::is_move_assignable_v<event_type>);

  // Check deleted constructors
  EXPECT_FALSE((std::is_constructible_v<event_type, int>));
  EXPECT_FALSE((std::is_constructible_v<event_type, std::nullptr_t>));

  // Check valid constructors
  EXPECT_TRUE((std::is_constructible_v<event_type, details_::deviceEvent_t>));
  EXPECT_TRUE((std::is_constructible_v<event_type, details_::event_wrap>));
}

// Test constexpr functionality
TEST_F(EventWrapTest, ConstexprFunctionality) {
  // Test constexpr constructor with raw event
  constexpr details_::deviceEvent_t null_event = nullptr;
  constexpr event_wrap const_event(null_event);
  EXPECT_FALSE(static_cast<bool>(const_event));
}

// Test duration type aliases
TEST_F(EventWrapTest, DurationTypeAliases) {
  // Verify duration type aliases are properly defined
  EXPECT_TRUE(
    (std::is_same_v<nanoSec, std::chrono::duration<float, std::nano>>));
  EXPECT_TRUE(
    (std::is_same_v<microSec, std::chrono::duration<float, std::micro>>));
  EXPECT_TRUE(
    (std::is_same_v<milliSec, std::chrono::duration<float, std::milli>>));
  EXPECT_TRUE((std::is_same_v<sec, std::chrono::duration<float>>));
}
