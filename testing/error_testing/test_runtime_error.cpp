#include "tests_common.hpp"

#include <gcxx/runtime/error/runtime_error.hpp>

using namespace gcxx;

class RuntimeErrorTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(RuntimeErrorTest, ThrowsGPURuntimeError) {
  // Test that throwGPUError throws a GPURuntimeError exception
  EXPECT_THROW(
    {
      details_::throwGPUError(details_::deviceError_t(1), "Test error message");
    },
    details_::GPURuntimeError
  );
}

TEST_F(RuntimeErrorTest, ExceptionContainsErrorCode) {
  try {
    details_::throwGPUError(details_::deviceError_t(42), "Test error");
    FAIL() << "Expected GPURuntimeError to be thrown";
  } catch (const details_::GPURuntimeError& e) {
    EXPECT_EQ(e.getErrorCode(), details_::deviceError_t(42));
  }
}

TEST_F(RuntimeErrorTest, ExceptionContainsMessage) {
  try {
    details_::throwGPUError(details_::deviceError_t(1), "Custom error message");
    FAIL() << "Expected GPURuntimeError to be thrown";
  } catch (const details_::GPURuntimeError& e) {
    std::string what_str(e.what());
    EXPECT_TRUE(what_str.find("Custom error message") != std::string::npos);
  }
}

TEST_F(RuntimeErrorTest, ExceptionIsDerivedFromStdRuntimeError) {
  try {
    details_::throwGPUError(details_::deviceError_t(1), "Test");
    FAIL() << "Expected GPURuntimeError to be thrown";
  } catch (const std::runtime_error& e) {
    // Should catch as std::runtime_error - test passes if we reach here
  } catch (...) {
    FAIL() << "Exception should be catchable as std::runtime_error";
  }
}

TEST_F(RuntimeErrorTest, ExceptionIsDerivedFromStdException) {
  try {
    details_::throwGPUError(details_::deviceError_t(1), "Test");
    FAIL() << "Expected GPURuntimeError to be thrown";
  } catch (const std::exception& e) {
    // Should catch as std::exception - test passes if we reach here
  } catch (...) {
    FAIL() << "Exception should be catchable as std::exception";
  }
}
