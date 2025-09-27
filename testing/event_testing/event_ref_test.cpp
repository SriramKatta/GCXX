#include <gtest/gtest.h>
#include <gcxx/runtime/event/event.hpp>
#include <gcxx/runtime/stream/stream_ref.hpp>

using namespace gcxx;

class EventTest : public ::testing::Test {
 protected:
  void SetUp() override { GCXX_SAFE_RUNTIME_CALL(StreamCreate, (&stream_)); }

  void TearDown() override { GCXX_SAFE_RUNTIME_CALL(StreamDestroy, (stream_)); }

  GCXX_RUNTIME_BACKEND(Stream_t) stream_{};
};

TEST_F(EventTest, ConstructAndDestroy) {
  {
    Event e;
    EXPECT_NE(e.get(), nullptr);
  }  // auto destroyed here
}

TEST_F(EventTest, CreateFactory) {
  auto e = Event::Create();
  EXPECT_NE(e.get(), nullptr);
}

TEST_F(EventTest, MoveConstructorTransfersOwnership) {
  Event e1;
  auto raw1 = e1.get();

  Event e2(std::move(e1));
  EXPECT_EQ(e1.get(), details_::INVALID_EVENT);
  EXPECT_EQ(e2.get(), raw1);
}

TEST_F(EventTest, MoveAssignmentTransfersOwnership) {
  Event e1;
  Event e2;
  auto raw1 = e1.get();

  e2 = std::move(e1);
  EXPECT_EQ(e1.get(), nullptr);
  EXPECT_EQ(e2.get(), raw1);
}

TEST_F(EventTest, ReleaseTransfersHandle) {
  Event e;
  auto raw = e.get();

  event_ref ref = e.release();
  EXPECT_EQ(e.get(), nullptr);
  EXPECT_EQ(ref.get(), raw);

  // Destroy manually since ownership transferred
  GCXX_SAFE_RUNTIME_CALL(EventDestroy, (raw));
}

TEST_F(EventTest, RecordAndElapsedTime) {
  Event start;
  Event end;
  stream_ref s(stream_);

  start.RecordInStream(s);
  GCXX_SAFE_RUNTIME_CALL(StreamSynchronize, (stream_));

  end.RecordInStream(s);
  GCXX_SAFE_RUNTIME_CALL(StreamSynchronize, (stream_));

  auto elapsed = Event::ElapsedTimeBetween(start, end);
  EXPECT_GE(elapsed.count(), 0.0f);
}
