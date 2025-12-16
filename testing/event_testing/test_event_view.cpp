#include "tests_common.hpp"

#include <gcxx/runtime/event/event.hpp>
#include <gcxx/runtime/event/event_view.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

using namespace gcxx;

class EventViewTest : public ::testing::Test {
 protected:
  void SetUp() override {
    GCXX_SAFE_RUNTIME_CALL(StreamCreate, "Failed to Create GPU Stream",
                           &stream_);
    GCXX_SAFE_RUNTIME_CALL(EventCreate, "Failed to Create GPU Event", &event_);
  }

  void TearDown() override {
    GCXX_SAFE_RUNTIME_CALL(EventDestroy, "Failed to Destroy GPU Event",
                           event_);
    GCXX_SAFE_RUNTIME_CALL(StreamDestroy, "Failed to Destroy GPU Stream",
                           stream_);
  }

  GCXX_RUNTIME_BACKEND(Stream_t) stream_{};
  GCXX_RUNTIME_BACKEND(Event_t) event_{};
};

TEST_F(EventViewTest, DefaultConstructor) {
  EventView view;
  EXPECT_EQ(view.get(), details_::INVALID_EVENT);
  EXPECT_FALSE(static_cast<bool>(view));
}

TEST_F(EventViewTest, ConstructFromRawEvent) {
  EventView view(event_);
  EXPECT_EQ(view.get(), event_);
  EXPECT_TRUE(static_cast<bool>(view));
}

TEST_F(EventViewTest, CopyConstructor) {
  EventView view1(event_);
  EventView view2(view1);
  EXPECT_EQ(view1.get(), view2.get());
  EXPECT_EQ(view2.get(), event_);
}

TEST_F(EventViewTest, ImplicitConversionToRaw) {
  EventView view(event_);
  GCXX_RUNTIME_BACKEND(Event_t) raw = view;
  EXPECT_EQ(raw, event_);
}

TEST_F(EventViewTest, GetMethod) {
  EventView view(event_);
  EXPECT_EQ(view.get(), event_);
}

TEST_F(EventViewTest, BoolConversionValidEvent) {
  EventView view(event_);
  EXPECT_TRUE(static_cast<bool>(view));
}

TEST_F(EventViewTest, BoolConversionInvalidEvent) {
  EventView view;
  EXPECT_FALSE(static_cast<bool>(view));
}

TEST_F(EventViewTest, EqualityOperatorSameEvent) {
  EventView view1(event_);
  EventView view2(event_);
  EXPECT_TRUE(view1 == view2);
}

TEST_F(EventViewTest, EqualityOperatorDifferentEvents) {
  GCXX_RUNTIME_BACKEND(Event_t) event2{};
  GCXX_SAFE_RUNTIME_CALL(EventCreate, "Failed to Create GPU Event", &event2);

  EventView view1(event_);
  EventView view2(event2);
  EXPECT_FALSE(view1 == view2);

  GCXX_SAFE_RUNTIME_CALL(EventDestroy, "Failed to Destroy GPU Event", event2);
}

TEST_F(EventViewTest, InequalityOperator) {
  GCXX_RUNTIME_BACKEND(Event_t) event2{};
  GCXX_SAFE_RUNTIME_CALL(EventCreate, "Failed to Create GPU Event", &event2);

  EventView view1(event_);
  EventView view2(event2);
  EXPECT_TRUE(view1 != view2);

  GCXX_SAFE_RUNTIME_CALL(EventDestroy, "Failed to Destroy GPU Event", event2);
}

TEST_F(EventViewTest, InequalityOperatorSameEvent) {
  EventView view1(event_);
  EventView view2(event_);
  EXPECT_FALSE(view1 != view2);
}

TEST_F(EventViewTest, RecordInStreamWithView) {
  EventView view(event_);
  StreamView s(stream_);

  view.RecordInStream(s);
  GCXX_SAFE_RUNTIME_CALL(StreamSynchronize, "Failed to Synchronize GPU Stream",
                         stream_);

  EXPECT_TRUE(view.HasOccurred());
}

TEST_F(EventViewTest, Synchronize) {
  EventView view(event_);
  StreamView s(stream_);

  view.RecordInStream(s);
  view.Synchronize();

  EXPECT_TRUE(view.HasOccurred());
}

TEST_F(EventViewTest, HasOccurredAfterRecord) {
  EventView view(event_);
  StreamView s(stream_);

  view.RecordInStream(s);
  GCXX_SAFE_RUNTIME_CALL(StreamSynchronize, "Failed to Synchronize GPU Stream",
                         stream_);

  EXPECT_TRUE(view.HasOccurred());
}

TEST_F(EventViewTest, ElapsedTimeSince) {
  GCXX_RUNTIME_BACKEND(Event_t) startEvent{};
  GCXX_RUNTIME_BACKEND(Event_t) endEvent{};
  GCXX_SAFE_RUNTIME_CALL(EventCreate, "Failed to Create GPU Event",
                         &startEvent);
  GCXX_SAFE_RUNTIME_CALL(EventCreate, "Failed to Create GPU Event", &endEvent);

  EventView start(startEvent);
  EventView end(endEvent);
  StreamView s(stream_);

  start.RecordInStream(s);
  GCXX_SAFE_RUNTIME_CALL(StreamSynchronize, "Failed to Synchronize GPU Stream",
                         stream_);

  end.RecordInStream(s);
  GCXX_SAFE_RUNTIME_CALL(StreamSynchronize, "Failed to Synchronize GPU Stream",
                         stream_);

  auto elapsed = end.ElapsedTimeSince(start);
  EXPECT_GE(elapsed.count(), 0.0f);

  GCXX_SAFE_RUNTIME_CALL(EventDestroy, "Failed to Destroy GPU Event",
                         startEvent);
  GCXX_SAFE_RUNTIME_CALL(EventDestroy, "Failed to Destroy GPU Event", endEvent);
}

TEST_F(EventViewTest, ElapsedTimeBetween) {
  GCXX_RUNTIME_BACKEND(Event_t) startEvent{};
  GCXX_RUNTIME_BACKEND(Event_t) endEvent{};
  GCXX_SAFE_RUNTIME_CALL(EventCreate, "Failed to Create GPU Event",
                         &startEvent);
  GCXX_SAFE_RUNTIME_CALL(EventCreate, "Failed to Create GPU Event", &endEvent);

  EventView start(startEvent);
  EventView end(endEvent);
  StreamView s(stream_);

  start.RecordInStream(s);
  GCXX_SAFE_RUNTIME_CALL(StreamSynchronize, "Failed to Synchronize GPU Stream",
                         stream_);

  end.RecordInStream(s);
  GCXX_SAFE_RUNTIME_CALL(StreamSynchronize, "Failed to Synchronize GPU Stream",
                         stream_);

  auto elapsed = EventView::ElapsedTimeBetween(start, end);
  EXPECT_GE(elapsed.count(), 0.0f);

  GCXX_SAFE_RUNTIME_CALL(EventDestroy, "Failed to Destroy GPU Event",
                         startEvent);
  GCXX_SAFE_RUNTIME_CALL(EventDestroy, "Failed to Destroy GPU Event", endEvent);
}

TEST_F(EventViewTest, ElapsedTimeWithDifferentDurationTypes) {
  GCXX_RUNTIME_BACKEND(Event_t) startEvent{};
  GCXX_RUNTIME_BACKEND(Event_t) endEvent{};
  GCXX_SAFE_RUNTIME_CALL(EventCreate, "Failed to Create GPU Event",
                         &startEvent);
  GCXX_SAFE_RUNTIME_CALL(EventCreate, "Failed to Create GPU Event", &endEvent);

  EventView start(startEvent);
  EventView end(endEvent);
  StreamView s(stream_);

  start.RecordInStream(s);
  GCXX_SAFE_RUNTIME_CALL(StreamSynchronize, "Failed to Synchronize GPU Stream",
                         stream_);

  end.RecordInStream(s);
  GCXX_SAFE_RUNTIME_CALL(StreamSynchronize, "Failed to Synchronize GPU Stream",
                         stream_);

  auto elapsedMs   = end.ElapsedTimeSince<milliSec>(start);
  auto elapsedUs   = end.ElapsedTimeSince<microSec>(start);
  auto elapsedNs   = end.ElapsedTimeSince<nanoSec>(start);
  auto elapsedSecs = end.ElapsedTimeSince<sec>(start);

  EXPECT_GE(elapsedMs.count(), 0.0f);
  EXPECT_GE(elapsedUs.count(), 0.0f);
  EXPECT_GE(elapsedNs.count(), 0.0f);
  EXPECT_GE(elapsedSecs.count(), 0.0f);

  GCXX_SAFE_RUNTIME_CALL(EventDestroy, "Failed to Destroy GPU Event",
                         startEvent);
  GCXX_SAFE_RUNTIME_CALL(EventDestroy, "Failed to Destroy GPU Event", endEvent);
}

TEST_F(EventViewTest, EventViewFromEvent) {
  Event e;
  EventView view = e.get();
  EXPECT_EQ(view.get(), e.get());
}

TEST_F(EventViewTest, MultipleViewsSameEvent) {
  EventView view1(event_);
  EventView view2(event_);
  EventView view3(view1);

  EXPECT_EQ(view1.get(), view2.get());
  EXPECT_EQ(view2.get(), view3.get());
  EXPECT_TRUE(view1 == view2);
  EXPECT_TRUE(view2 == view3);
}
