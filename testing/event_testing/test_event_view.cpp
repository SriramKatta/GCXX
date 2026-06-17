// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include "tests_common.hpp"

#include <type_traits>

#include <gcxx/runtime/event/event.hpp>
#include <gcxx/runtime/event/event_view.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

using namespace gcxx;

class EventViewTest : public ::testing::Test {
 protected:
  void SetUp() override {
    m_stream = driver::streamCreateWithPriority(
      static_cast<details_::flag_t>(flags::streamType::SyncWithNull), 0);
    m_event = driver::eventCreateWithFlags(
      static_cast<details_::flag_t>(flags::eventCreate::None));
  }

  void TearDown() override {
    driver::eventDestroy(m_event);
    driver::streamDestroy(m_stream);
  }

  driver::deviceStream_t m_stream{driver::NULL_STREAM};
  driver::deviceEvent_t m_event{driver::INVALID_EVENT};
};

TEST_F(EventViewTest, DefaultConstructor) {
  EventView view;
  EXPECT_EQ(view.getRawEvent(), driver::INVALID_EVENT);
  EXPECT_FALSE(static_cast<bool>(view));
}

TEST_F(EventViewTest, ConstructFromRawEvent) {
  EventView view(m_event);
  EXPECT_EQ(view.getRawEvent(), m_event);
  EXPECT_TRUE(static_cast<bool>(view));
}

TEST_F(EventViewTest, CopyConstructor) {
  EventView view1(m_event);
  EventView view2(view1);
  EXPECT_EQ(view1.getRawEvent(), view2.getRawEvent());
  EXPECT_EQ(view2.getRawEvent(), m_event);
}

TEST_F(EventViewTest, CopyAssignmentRebindsToSameEvent) {
  driver::deviceEvent_t event2{driver::INVALID_EVENT};
  event2 = driver::eventCreateWithFlags(
    static_cast<details_::flag_t>(flags::eventCreate::None));

  EventView source(m_event);
  EventView target(event2);
  target = source;

  EXPECT_EQ(target.getRawEvent(), m_event);
  EXPECT_TRUE(target == source);

  driver::eventDestroy(event2);
}

TEST_F(EventViewTest, ImplicitConversionToRaw) {
  EventView view(m_event);
  driver::deviceEvent_t raw = view;
  EXPECT_EQ(raw, m_event);
}

TEST_F(EventViewTest, getRawEventMethod) {
  EventView view(m_event);
  EXPECT_EQ(view.getRawEvent(), m_event);
}

TEST_F(EventViewTest, BoolConversionValidEvent) {
  EventView view(m_event);
  EXPECT_TRUE(static_cast<bool>(view));
}

TEST_F(EventViewTest, BoolConversionInvalidEvent) {
  EventView view;
  EXPECT_FALSE(static_cast<bool>(view));
}

TEST(EventViewCompileTimeTest, RejectsAmbiguousIntegralAndNullConstruction) {
  static_assert(!std::is_constructible_v<EventView, int>);
  static_assert(!std::is_constructible_v<EventView, std::nullptr_t>);
}

TEST(EventViewDurationTest, ConvertsMillisecondsToSupportedDurations) {
  EXPECT_FLOAT_EQ(ConvertDuration<milliSec>(1.5F).count(), 1.5F);
  EXPECT_FLOAT_EQ(ConvertDuration<microSec>(1.5F).count(), 1500.0F);
  EXPECT_FLOAT_EQ(ConvertDuration<nanoSec>(1.5F).count(), 1500000.0F);
  EXPECT_FLOAT_EQ(ConvertDuration<sec>(1.5F).count(), 0.0015F);
}

TEST_F(EventViewTest, EqualityOperatorSameEvent) {
  EventView view1(m_event);
  EventView view2(m_event);
  EXPECT_TRUE(view1 == view2);
}

TEST_F(EventViewTest, EqualityOperatorDifferentEvents) {
  driver::deviceEvent_t event2{driver::INVALID_EVENT};
  event2 = driver::eventCreateWithFlags(
    static_cast<details_::flag_t>(flags::eventCreate::None));

  EventView view1(m_event);
  EventView view2(event2);
  EXPECT_FALSE(view1 == view2);

  driver::eventDestroy(event2);
}

TEST_F(EventViewTest, InequalityOperator) {
  driver::deviceEvent_t event2{driver::INVALID_EVENT};
  event2 = driver::eventCreateWithFlags(
    static_cast<details_::flag_t>(flags::eventCreate::None));

  EventView view1(m_event);
  EventView view2(event2);
  EXPECT_TRUE(view1 != view2);

  driver::eventDestroy(event2);
}

TEST_F(EventViewTest, InequalityOperatorSameEvent) {
  EventView view1(m_event);
  EventView view2(m_event);
  EXPECT_FALSE(view1 != view2);
}

TEST_F(EventViewTest, RecordInStreamWithView) {
  EventView view(m_event);
  StreamView s(m_stream);

  view.RecordInStream(s);
  driver::streamSynchronize(m_stream);

  EXPECT_TRUE(view.HasOccurred());
}

TEST_F(EventViewTest, RecordInDefaultStream) {
  EventView view(m_event);

  view.RecordInStream();
  view.Synchronize();

  EXPECT_TRUE(view.HasOccurred());
}

TEST_F(EventViewTest, Synchronize) {
  EventView view(m_event);
  StreamView s(m_stream);

  view.RecordInStream(s);
  view.Synchronize();

  EXPECT_TRUE(view.HasOccurred());
}

TEST_F(EventViewTest, HasOccurredAfterRecord) {
  EventView view(m_event);
  StreamView s(m_stream);

  view.RecordInStream(s);
  driver::streamSynchronize(m_stream);

  EXPECT_TRUE(view.HasOccurred());
}

TEST_F(EventViewTest, ElapsedTimeSince) {
  driver::deviceEvent_t startEvent{driver::INVALID_EVENT};
  driver::deviceEvent_t endEvent{driver::INVALID_EVENT};
  startEvent = driver::eventCreateWithFlags(
    static_cast<details_::flag_t>(flags::eventCreate::None));
  endEvent = driver::eventCreateWithFlags(
    static_cast<details_::flag_t>(flags::eventCreate::None));

  EventView start(startEvent);
  EventView end(endEvent);
  StreamView s(m_stream);

  start.RecordInStream(s);
  driver::streamSynchronize(m_stream);

  end.RecordInStream(s);
  driver::streamSynchronize(m_stream);

  auto elapsed = end.ElapsedTimeSince(start);
  EXPECT_GE(elapsed.count(), 0.0f);

  driver::eventDestroy(startEvent);
  driver::eventDestroy(endEvent);
}

TEST_F(EventViewTest, ElapsedTimeBetween) {
  driver::deviceEvent_t startEvent{driver::INVALID_EVENT};
  driver::deviceEvent_t endEvent{driver::INVALID_EVENT};
  startEvent = driver::eventCreateWithFlags(
    static_cast<details_::flag_t>(flags::eventCreate::None));
  endEvent = driver::eventCreateWithFlags(
    static_cast<details_::flag_t>(flags::eventCreate::None));

  EventView start(startEvent);
  EventView end(endEvent);
  StreamView s(m_stream);

  start.RecordInStream(s);
  driver::streamSynchronize(m_stream);

  end.RecordInStream(s);
  driver::streamSynchronize(m_stream);

  auto elapsed = EventView::ElapsedTimeBetween(start, end);
  EXPECT_GE(elapsed.count(), 0.0f);

  driver::eventDestroy(startEvent);
  driver::eventDestroy(endEvent);
}

TEST_F(EventViewTest, ElapsedTimeWithDifferentDurationTypes) {
  driver::deviceEvent_t startEvent{driver::INVALID_EVENT};
  driver::deviceEvent_t endEvent{driver::INVALID_EVENT};
  startEvent = driver::eventCreateWithFlags(
    static_cast<details_::flag_t>(flags::eventCreate::None));
  endEvent = driver::eventCreateWithFlags(
    static_cast<details_::flag_t>(flags::eventCreate::None));

  EventView start(startEvent);
  EventView end(endEvent);
  StreamView s(m_stream);

  start.RecordInStream(s);
  driver::streamSynchronize(m_stream);

  end.RecordInStream(s);
  driver::streamSynchronize(m_stream);

  auto elapsedMs   = end.ElapsedTimeSince<milliSec>(start);
  auto elapsedUs   = end.ElapsedTimeSince<microSec>(start);
  auto elapsedNs   = end.ElapsedTimeSince<nanoSec>(start);
  auto elapsedSecs = end.ElapsedTimeSince<sec>(start);

  EXPECT_GE(elapsedMs.count(), 0.0f);
  EXPECT_GE(elapsedUs.count(), 0.0f);
  EXPECT_GE(elapsedNs.count(), 0.0f);
  EXPECT_GE(elapsedSecs.count(), 0.0f);

  driver::eventDestroy(startEvent);
  driver::eventDestroy(endEvent);
}

TEST_F(EventViewTest, EventViewFromEvent) {
  Event e;
  EventView view = e.getRawEvent();
  EXPECT_EQ(view.getRawEvent(), e.getRawEvent());
}

TEST_F(EventViewTest, MultipleViewsSameEvent) {
  EventView view1(m_event);
  EventView view2(m_event);
  EventView view3(view1);

  EXPECT_EQ(view1.getRawEvent(), view2.getRawEvent());
  EXPECT_EQ(view2.getRawEvent(), view3.getRawEvent());
  EXPECT_TRUE(view1 == view2);
  EXPECT_TRUE(view2 == view3);
}
