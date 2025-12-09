#include "tests_common.hpp"

#include <gcxx/runtime/error/runtime_error.hpp>

using namespace gcxx;

class RuntimeErrorTest : public ::testing::Test {};

TEST_F(RuntimeErrorTest, ThrowsGPURuntimeError) {
  // Test that throwGPUError actually throws an exception
  EXPECT_THROW(
    {
      details_::throwGPUError(
        static_cast<details_::deviceError_t>(1),
        "Test error message"
      );
    },
    details_::gpu_runtime_error
  );
}

TEST_F(RuntimeErrorTest, ExceptionInheritsFromStdRuntimeError) {
  // Test that gpu_runtime_error can be caught as std::runtime_error
  try {
    details_::throwGPUError(
      static_cast<details_::deviceError_t>(1),
      "Test error message"
    );
    FAIL() << "Expected gpu_runtime_error to be thrown";
  } catch (const std::runtime_error& e) {
    // Successfully caught as std::runtime_error
    EXPECT_TRUE(std::string(e.what()).find("Test error message") != std::string::npos);
  } catch (...) {
    FAIL() << "Caught unexpected exception type";
  }
}

TEST_F(RuntimeErrorTest, ExceptionContainsErrorCode) {
  // Test that the exception stores the error code
  try {
    const details_::deviceError_t test_code = static_cast<details_::deviceError_t>(42);
    details_::throwGPUError(test_code, "Test error");
    FAIL() << "Expected exception to be thrown";
  } catch (const details_::gpu_runtime_error& e) {
    EXPECT_EQ(e.error_code(), static_cast<details_::deviceError_t>(42));
  }
}

TEST_F(RuntimeErrorTest, ExceptionMessageContainsUserMessage) {
  // Test that the exception message contains the user-provided message
  try {
    details_::throwGPUError(
      static_cast<details_::deviceError_t>(1),
      "Custom error message"
    );
    FAIL() << "Expected exception to be thrown";
  } catch (const std::runtime_error& e) {
    std::string message = e.what();
    EXPECT_TRUE(message.find("Custom error message") != std::string::npos);
  }
}

TEST_F(RuntimeErrorTest, ExceptionMessageContainsErrorCode) {
  // Test that the exception message contains error code information
  try {
    details_::throwGPUError(
      static_cast<details_::deviceError_t>(99),
      "Test"
    );
    FAIL() << "Expected exception to be thrown";
  } catch (const std::runtime_error& e) {
    std::string message = e.what();
    EXPECT_TRUE(message.find("99") != std::string::npos);
  }
}

TEST_F(RuntimeErrorTest, HandlesNullMessage) {
  // Test that the exception handles null message gracefully
  try {
    details_::throwGPUError(
      static_cast<details_::deviceError_t>(1),
      nullptr
    );
    FAIL() << "Expected exception to be thrown";
  } catch (const std::runtime_error& e) {
    std::string message = e.what();
    // Should contain "(no message)" instead of crashing
    EXPECT_TRUE(message.find("(no message)") != std::string::npos);
  }
}
