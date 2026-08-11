// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include "tests_common.hpp"

#include <utility>

#include <gcxx/runtime/event/event.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

// raw_handle_type contract (see tests_common.hpp).
GCXX_ASSERT_RAW_HANDLE(Event, gcxx::driver::deviceEvent_t);

using namespace gcxx;

class EventTest : public ::testing::Test {
 protected:
  void SetUp() override {
    m_stream = driver::streamCreateWithPriority(
      static_cast<details_::flag_t>(flags::streamType::SyncWithNull), 0);
  }

  void TearDown() override { driver::streamDestroy(m_stream); }

  driver::deviceStream_t m_stream{driver::NULL_STREAM};
};

TEST_F(EventTest, ConstructAndDestroy) {
  {
    Event e;
    EXPECT_NE(e.getRawHandle(), nullptr);
  }  // auto destroyed here
}

TEST_F(EventTest, CreateFactory) {
  auto e = Event();
  EXPECT_NE(e.getRawHandle(), nullptr);
}

TEST_F(EventTest, CreateWithFlagsProducesUsableEvent) {
  auto e = Event(flags::eventCreate::blockingSync);
  EXPECT_NE(e.getRawHandle(), driver::INVALID_EVENT);

  e.RecordInStream();
  e.Synchronize();

  EXPECT_TRUE(e.HasOccurred());
}

TEST_F(EventTest, MoveConstructorTransfersOwnership) {
  Event e1;
  auto raw1 = e1.getRawHandle();

  Event e2(std::move(e1));
  EXPECT_EQ(e1.getRawHandle(), driver::INVALID_EVENT);
  EXPECT_EQ(e2.getRawHandle(), raw1);
}

TEST_F(EventTest, MoveAssignmentTransfersOwnership) {
  Event e1;
  Event e2;
  auto raw1 = e1.getRawHandle();

  e2 = std::move(e1);
  EXPECT_EQ(e1.getRawHandle(), gcxx::driver::INVALID_EVENT);
  EXPECT_EQ(e2.getRawHandle(), raw1);
}

TEST_F(EventTest, ReleaseTransfersHandle) {
  Event e;
  auto raw = e.getRawHandle();

  EventView ref = e.Release();
  EXPECT_EQ(e.getRawHandle(), gcxx::driver::INVALID_EVENT);
  EXPECT_EQ(ref.getRawHandle(), raw);

  // Destroy manually since ownership transferred
  driver::eventDestroy(raw);
}

TEST_F(EventTest, ReleasedHandleRemainsUsableThroughView) {
  Event e;
  auto raw      = e.getRawHandle();
  EventView ref = e.Release();

  ref.RecordInStream();
  ref.Synchronize();

  EXPECT_TRUE(ref.HasOccurred());
  driver::eventDestroy(raw);
}

TEST_F(EventTest, RecordAndElapsedTime) {
  Event start;
  Event end;
  StreamView s(m_stream);

  start.RecordInStream(s);
  driver::streamSynchronize(m_stream);

  end.RecordInStream(s);
  driver::streamSynchronize(m_stream);

  auto elapsed = Event::ElapsedTimeBetween(start, end);
  EXPECT_GE(elapsed.count(), 0.0f);
}

TEST_F(EventTest, StreamViewRecordEventReturnsRecordedEvent) {
  StreamView s(m_stream);

  auto event = s.RecordEvent();
  s.Synchronize();

  EXPECT_TRUE(event.HasOccurred());
}

TEST_F(EventTest, MoveAssignmentSelfMoveKeepsOwnership) {
  Event e;
  const auto raw = e.getRawHandle();

  e = std::move(e);

  EXPECT_EQ(e.getRawHandle(), raw);
}
