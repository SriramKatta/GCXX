#include "tests_common.hpp"


#include <gcxx/runtime/event/event.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

using namespace gcxx;

class EventTest : public ::testing::Test {
 protected:
  void SetUp() override {
    GCXX_SAFE_RUNTIME_CALL(StreamCreate, "Failed to Create GPU Stream",
                           &stream_);
  }

  void TearDown() override {
    GCXX_SAFE_RUNTIME_CALL(StreamDestroy, "Failed to Destroy GPU Stream",
                           stream_);
  }

  GCXX_RUNTIME_BACKEND(Stream_t) stream_{};
};

TEST_F(EventTest, ConstructAndDestroy) {
  {
    Event e;
    EXPECT_NE(e.getRawEvent(), nullptr);
  }  // auto destroyed here
}

TEST_F(EventTest, CreateFactory) {
  auto e = Event::Create();
  EXPECT_NE(e.getRawEvent(), nullptr);
}

TEST_F(EventTest, MoveConstructorTransfersOwnership) {
  Event e1;
  auto raw1 = e1.getRawEvent();

  Event e2(std::move(e1));
  EXPECT_EQ(e1.getRawEvent(), details_::INVALID_EVENT);
  EXPECT_EQ(e2.getRawEvent(), raw1);
}

TEST_F(EventTest, MoveAssignmentTransfersOwnership) {
  Event e1;
  Event e2;
  auto raw1 = e1.getRawEvent();

  e2 = std::move(e1);
  EXPECT_EQ(e1.getRawEvent(), nullptr);
  EXPECT_EQ(e2.getRawEvent(), raw1);
}

TEST_F(EventTest, ReleaseTransfersHandle) {
  Event e;
  auto raw = e.getRawEvent();

  EventView ref = e.release();
  EXPECT_EQ(e.getRawEvent(), nullptr);
  EXPECT_EQ(ref.getRawEvent(), raw);

  // Destroy manually since ownership transferred
  GCXX_SAFE_RUNTIME_CALL(EventDestroy, "Failed to Destroy GPU Event", raw);
}

TEST_F(EventTest, RecordAndElapsedTime) {
  Event start;
  Event end;
  StreamView s(stream_);

  start.RecordInStream(s);
  GCXX_SAFE_RUNTIME_CALL(StreamSynchronize, "Failed to Synchronize GPU Stream",
                         stream_);

  end.RecordInStream(s);
  GCXX_SAFE_RUNTIME_CALL(StreamSynchronize, "Failed to Synchronize GPU Stream",
                         stream_);

  auto elapsed = Event::ElapsedTimeBetween(start, end);
  EXPECT_GE(elapsed.count(), 0.0f);
}
