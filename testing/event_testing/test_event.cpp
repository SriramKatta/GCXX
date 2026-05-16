// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include "tests_common.hpp"

#include <utility>

#include <gcxx/runtime/event/event.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

using namespace gcxx;

class EventTest : public ::testing::Test {
 protected:
  void SetUp() override {
    stream_ = driver::streamCreateWithPriority(
      static_cast<details_::flag_t>(flags::streamType::SyncWithNull), 0);
  }

  void TearDown() override { driver::streamDestroy(stream_); }

  driver::deviceStream_t stream_{driver::NULL_STREAM};
};

TEST_F(EventTest, ConstructAndDestroy) {
  {
    Event e;
    EXPECT_NE(e.getRawEvent(), nullptr);
  }  // auto destroyed here
}

TEST_F(EventTest, CreateFactory) {
  auto e = Event();
  EXPECT_NE(e.getRawEvent(), nullptr);
}

TEST_F(EventTest, CreateWithFlagsProducesUsableEvent) {
  auto e = Event(flags::eventCreate::blockingSync);
  EXPECT_NE(e.getRawEvent(), driver::INVALID_EVENT);

  e.RecordInStream();
  e.Synchronize();

  EXPECT_TRUE(e.HasOccurred());
}

TEST_F(EventTest, MoveConstructorTransfersOwnership) {
  Event e1;
  auto raw1 = e1.getRawEvent();

  Event e2(std::move(e1));
  EXPECT_EQ(e1.getRawEvent(), driver::INVALID_EVENT);
  EXPECT_EQ(e2.getRawEvent(), raw1);
}

TEST_F(EventTest, MoveAssignmentTransfersOwnership) {
  Event e1;
  Event e2;
  auto raw1 = e1.getRawEvent();

  e2 = std::move(e1);
  EXPECT_EQ(e1.getRawEvent(), gcxx::driver::INVALID_EVENT);
  EXPECT_EQ(e2.getRawEvent(), raw1);
}

TEST_F(EventTest, ReleaseTransfersHandle) {
  Event e;
  auto raw = e.getRawEvent();

  EventView ref = e.Release();
  EXPECT_EQ(e.getRawEvent(), gcxx::driver::INVALID_EVENT);
  EXPECT_EQ(ref.getRawEvent(), raw);

  // Destroy manually since ownership transferred
  driver::eventDestroy(raw);
}

TEST_F(EventTest, ReleasedHandleRemainsUsableThroughView) {
  Event e;
  auto raw      = e.getRawEvent();
  EventView ref = e.Release();

  ref.RecordInStream();
  ref.Synchronize();

  EXPECT_TRUE(ref.HasOccurred());
  driver::eventDestroy(raw);
}

TEST_F(EventTest, RecordAndElapsedTime) {
  Event start;
  Event end;
  StreamView s(stream_);

  start.RecordInStream(s);
  driver::streamSynchronize(stream_);

  end.RecordInStream(s);
  driver::streamSynchronize(stream_);

  auto elapsed = Event::ElapsedTimeBetween(start, end);
  EXPECT_GE(elapsed.count(), 0.0f);
}

TEST_F(EventTest, StreamViewRecordEventReturnsRecordedEvent) {
  StreamView s(stream_);

  auto event = s.RecordEvent();
  s.Synchronize();

  EXPECT_TRUE(event.HasOccurred());
}

TEST_F(EventTest, MoveAssignmentSelfMoveKeepsOwnership) {
  Event e;
  const auto raw = e.getRawEvent();

  e = std::move(e);

  EXPECT_EQ(e.getRawEvent(), raw);
}
